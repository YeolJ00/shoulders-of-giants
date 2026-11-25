#!/bin/bash
# The fit at one material (2026-11-19): the teacher's next frame is set largely by the material (another episode's
# material moves it 0.90 of constant velocity's error on free cloth) while the network's prediction moves 0.05-0.06
# when its given material is hidden or swapped. Is the geometry -> force map learnable when the material never changes?
# One fully empty GPU (no one's process on it, checked twice a minute apart):
#   1. c1m_test (256 held-out) and c1m (1,024), the c1 generator with every cloth at the prior's centre
#      (w 0.16 kg/m^2, Y 3,000 N/m, bending length 2.7 cm, mu 0.35)
#   2. the control (ck1: all materials, positions only) scored on c1m_test while c1m is generated
#   3. the ck1 recipe on c1m, 8,000 steps (~5 h); scored on the same c1m_test windows at 4,000 and 7,999
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
M=0.16,3000,0.027,0.35
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  for g in $(empty_gpus); do
    sleep 60
    if empty_gpus | grep -qx "$g"; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
echo "[onemat] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L scripts/l1_generate_cloth.py --out data/datasets/c1m_test --episodes 256 --worlds 64 --split test --seed 23 --material $M \
  > outputs/lsim/gen_c1m_test.log 2>&1 || { echo "[onemat] c1m_test failed"; exit 1; }
echo "[onemat] c1m_test done $(date -u)"
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1m_test --n 300 --tag ck1_on_c1m \
  > outputs/lsim/where_ck1_on_c1m.log 2>&1 &
$L scripts/l1_generate_cloth.py --out data/datasets/c1m --episodes 1024 --worlds 64 --seed 22 --material $M \
  > outputs/lsim/gen_c1m.log 2>&1 || { echo "[onemat] c1m failed"; exit 1; }
wait
echo "[onemat] c1m done $(date -u)"
$L -m lsim.train.cloth_train --data data/datasets/c1m --val "data/datasets/c1m_test/shard_*.npz" \
  --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 \
  --prob_slots --unc --accel --layer 2 --edge 1 --steps 8000 --keep --out outputs/lsim/cloth_ck1_onemat \
  > outputs/lsim/cloth_ck1_onemat.log 2>&1 || { echo "[onemat] training failed"; exit 1; }
echo "[onemat] trained $(date -u)"
for s in 4000 7999; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_onemat/step$s.pt --data c1m_test --n 300 --tag onemat_$s \
    > outputs/lsim/where_onemat_$s.log 2>&1
done
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step4000.pt --data c1m_test --n 300 --tag ck1_on_c1m_4000 \
  > outputs/lsim/where_ck1_on_c1m_4000.log 2>&1
echo "[onemat] scored $(date -u)"
