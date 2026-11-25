#!/bin/bash
# Model size at equal windows (2026-11-20): the 509M network (4 GPUs x 4 windows, 12k steps, lr 3e-4, same data and
# recipe otherwise) at steps 4,000 / 8,000 / 11,999 = 64k / 128k / 192k windows, on the same 300 c1_test windows
# (probe R 6) as the 64M curve (q64 at 8,000 / 16,000 / 23,999 = the same windows seen). Its trainer val (0.53 mm vs
# q64's 0.73) and the earlier fit curve (0.69 vs 0.68 mm, every slot given) disagree; this settles it on our metric.
# One fully empty GPU (no one's process on it, checked twice a minute apart).
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
echo "[q509] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
for s in 4000 8000 11999; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_q509/step$s.pt --data c1_test --n 300 --probe_R 6 \
    --tag q509_${s}_on_c1_R6 > outputs/lsim/where_q509_${s}_on_c1_R6.log 2>&1 &
done
wait
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_q509/step11999.pt --data c1_trainsub --n 300 --probe_R 6 \
  --tag q509_on_train > outputs/lsim/where_q509_on_train.log 2>&1
echo "[q509] done $(date -u)"
