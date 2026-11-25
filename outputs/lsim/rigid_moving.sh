#!/bin/bash
# Rigid track, one v0 episode in the view it is scored in (2026-11-23; scratch rigid_fixed.py --moving --K 1): every
# window whose next frame moves a rigid body (56 starts) + 12 still ones, each exactly the rollout's view (8 given
# frames, one step), FIXED, batch 8, 4,000 steps (~440 views a window); no augmentation, noise or pushforward. Then
# the small-fit score (one frame over constant velocity, replay from frame 8) and the per-body-frame breakdown.
# On four fixed 8-step windows the network fits what it has a contact edge for to 0.05-0.3 mm; the floor was a landing
# from beyond the 3 cm contact radius. Arms: base | grip (--grip_force) | both (--r_contact 0.05 --grip_force).
#   $1 = name, $2 = GPU (checked empty twice a minute apart), $3.. = flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
n=$1; g=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[moving] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L /tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/rigid_fixed.py --moving --K 1 --batch 8 --steps 4000 --out outputs/lsim/rigid_moving_$n.pt "$@" > outputs/lsim/rigid_moving_$n.log 2>&1
echo "[moving] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/rigid_moving_$n.pt --episodes 1 > outputs/lsim/small_fit_rigid_moving_$n.log 2>&1
CUDA_VISIBLE_DEVICES=$g $L /tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/rigid_where.py outputs/lsim/rigid_moving_$n.pt > outputs/lsim/rigid_moving_${n}_where.log 2>&1
echo "[moving] $n scored ($?) $(date -u)"
