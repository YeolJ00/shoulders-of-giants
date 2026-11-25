#!/bin/bash
# Rigid track, four FIXED windows of one v0 episode (scratch rigid_fixed.py; 2026-11-23): the rigid network trained on
# only these (1,500 steps, no augmentation / noise / pushforward) stalls at first step 0.34, all 8 steps 0.13 of
# constant velocity. Is the floor what the network cannot see or push on? 2 x 2:
#   base    r_contact 0.03, no learned force on gripper edges (the model as built)
#   reach   r_contact 0.07 (v0 bodies travel up to 6 cm a frame; edges are built at the frame's start)
#   grip    learned force on gripper edges
#   both
#   $1 = name, $2 = GPU (checked empty twice a minute apart), $3.. = flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
n=$1; g=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[fixed] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L /tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/rigid_fixed.py --n 4 --steps 1500 --out outputs/lsim/rigid_fixed4_$n.pt "$@" > outputs/lsim/rigid_fixed4_$n.log 2>&1
echo "[fixed] $n done ($?) $(date -u)"
