#!/bin/bash
# The material-blind floor (2026-11-20, scripts/l1_teacher_blind.py): the teacher from 3 frame positions with the next
# frame averaged over 15 other episodes' materials -- the best one-frame error a predictor that cannot use the material
# can reach -- against the networks' plateau at ~0.65-0.67. One fully empty GPU (checked twice a minute apart).
cd /home/nas5/jooyeolyun/repos/simulation-assestization/scripts
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
echo "[blind] GPU $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L l1_teacher_blind.py
echo "[blind] done $(date -u)"
