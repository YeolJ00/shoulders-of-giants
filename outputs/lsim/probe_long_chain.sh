#!/bin/bash
# Is the dense local fit limited by training length (2026-11-20; l1_local_probe_scale.py --long_test)? One fully empty GPU
# (no one's process on it, checked twice a minute apart), ~30 min.
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
echo "[long] GPU $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L scripts/l1_local_probe_scale.py --long_test --steps 120000 --cache outputs/lsim/probe_scale_ep450 > outputs/lsim/probe_long.log 2>&1 || { echo "[long] failed"; exit 1; }
echo "[long] done $(date -u)"
