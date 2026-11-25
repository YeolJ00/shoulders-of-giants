#!/bin/bash
# The replay floor (2026-11-22): scripts/l1_teacher_chaos.py -- the teacher restarted from one imperfect frame of its
# own episode, run exactly to the end. GPU $1 once it is empty twice a minute apart.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[chaos] GPU $g $(date -u)"
cd scripts && CUDA_VISIBLE_DEVICES=$g $L l1_teacher_chaos.py > ../outputs/lsim/teacher_chaos_run.log 2>&1
echo "[chaos] done ($?) $(date -u)"
