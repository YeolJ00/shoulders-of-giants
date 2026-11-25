#!/bin/bash
# Move the error breakdown to a GPU as soon as one is fully empty (checked twice a minute apart); the CPU copy is
# limited to one core here (~40 s per window).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
while true; do
  grep -q "squared error" outputs/lsim/error_where.log && exit 0
  for g in $(empty_gpus); do
    sleep 60
    if empty_gpus | grep -qx "$g"; then
      pid=$(ps aux | grep "[l]2_error_where.py --ckpt" | awk '{print $2}')
      [ -n "$pid" ] && kill $pid
      echo "[where] on GPU $g $(date -u)"
      CUDA_VISIBLE_DEVICES=$g /home/jooyeolyun/anaconda3/envs/lsim/bin/python scripts/l2_error_where.py --ckpt outputs/lsim/cloth_q64/step23999.pt --n 300 > outputs/lsim/error_where.log 2>&1
      exit 0
    fi
  done
  sleep 120
done
