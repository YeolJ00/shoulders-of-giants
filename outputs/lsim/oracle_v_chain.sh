#!/bin/bash
# The oracle-velocity test (2026-11-19): does the network fit far better when handed the teacher's true velocity?
# One fully empty GPU (no one's process on it, checked twice a minute apart):
#   1. c1v_test (256 held-out episodes) and c1v (1,024 episodes), same generator as c1 + the teacher's velocities
#   2. the control (the ck1 recipe on c1, positions only) scored on c1v_test, while c1v is generated
#   3. the ck1 recipe + --oracle_v on c1v, 8,000 steps (~5 h)
#   4. both scored on the same c1v_test windows at steps 4,000 and 7,999
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
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
echo "[oracle] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L scripts/l1_generate_cloth.py --out data/datasets/c1v_test --episodes 256 --worlds 64 --split test --seed 21 --record_v \
  > outputs/lsim/gen_c1v_test.log 2>&1 || { echo "[oracle] c1v_test failed"; exit 1; }
echo "[oracle] c1v_test done $(date -u)"
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1v_test --n 300 --tag ck1_on_c1v \
  > outputs/lsim/where_ck1_on_c1v.log 2>&1 &
$L scripts/l1_generate_cloth.py --out data/datasets/c1v --episodes 1024 --worlds 64 --seed 20 --record_v \
  > outputs/lsim/gen_c1v.log 2>&1 || { echo "[oracle] c1v failed"; exit 1; }
wait
echo "[oracle] c1v done $(date -u)"
$L -m lsim.train.cloth_train --data data/datasets/c1v --val "data/datasets/c1v_test/shard_*.npz" --oracle_v \
  --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 \
  --prob_slots --unc --accel --layer 2 --edge 1 --steps 8000 --keep --out outputs/lsim/cloth_ck1_oracle \
  > outputs/lsim/cloth_ck1_oracle.log 2>&1 || { echo "[oracle] training failed"; exit 1; }
echo "[oracle] trained $(date -u)"
for s in 4000 7999; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_oracle/step$s.pt --data c1v_test --n 300 --tag oracle_v_$s \
    > outputs/lsim/where_oracle_v_$s.log 2>&1
done
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step4000.pt --data c1v_test --n 300 --tag ck1_on_c1v_4000 \
  > outputs/lsim/where_ck1_on_c1v_4000.log 2>&1
echo "[oracle] scored $(date -u)"
