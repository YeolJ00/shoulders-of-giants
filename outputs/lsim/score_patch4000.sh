#!/bin/bash
# Score the stencil run's step-4000 checkpoint on ck1's c1v_test windows (2026-11-20), early; one fully empty GPU
# (no one's process on it, checked twice a minute apart), ~4 min.
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
echo "[s4000] GPU $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_patch4/step4000.pt --data c1v_test --n 300 --tag patch4_4000 > outputs/lsim/where_patch4_4000.log 2>&1 || { echo "[s4000] failed"; exit 1; }
echo "[s4000] done $(date -u)"
