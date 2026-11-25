#!/bin/bash
# The rigid one-episode test without the training noise (2026-11-23). rigid_fit_1 (v1's recipe as it is) fits its one
# episode to 0.597 of constant velocity over one frame and rigid_fit_1_plainb (--plain) to 0.358 with a 2-3 mm error on
# every moving body-frame whatever its size -- but l2_overfit's --plain switches off the augmentation, the hidden slots
# and the history noise only: both runs kept the chunk noise (--noise 0.003,0.015: 3 mm and 0.015 rad a frame on the
# frames the network reads) and the 6-frame pushforward. A 3 mm input noise on bodies that move 0.1-5 mm a frame is a
# floor by construction. Two arms, 6,000 steps each, scored by l2_small_fit_rigid.py:
#   $2 = rigid_fit_1_clean      --plain --noise 0,0 --pf 0   nothing between the network and the frames
#   $2 = rigid_fit_1_aug        --noise 0,0 --pf 0           the augmentation and hidden slots on, no noise
#   $1 = GPU (empty twice a minute apart), $3.. = the arm's flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rigidc] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1 --steps 6000 --pair_norm --coulomb \
  --mix --noise 0,0 --pf 0 "$@" --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[rigidc] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 1 > outputs/lsim/small_fit_$n.log 2>&1
echo "[rigidc] $n scored ($?) $(date -u)"
