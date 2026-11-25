#!/bin/bash
# The frame-influence test on one fully empty GPU (checked twice a minute apart).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
e(){ busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u); nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done; }
while true; do
  for g in $(e); do
    sleep 60
    if e | grep -qx "$g"; then
      echo "[influence] GPU $g $(date -u)"
      CUDA_VISIBLE_DEVICES=$g /home/jooyeolyun/anaconda3/envs/lsim/bin/python scripts/l1_frame_influence.py "$@"
      echo "[influence] done $(date -u)"
      exit 0
    fi
  done
  sleep 120
done
