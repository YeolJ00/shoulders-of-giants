#!/bin/bash
# Data x training for the plain rigid transformer: the fourth cell (2026-11-23). rigid_direct_levers.sh's data arm
# (1,024 episodes, 12,000 steps) ended at 0.568 on the 32 held-out episodes of shard 8, where the 128-episode network
# at the same steps scores 0.539 (+0.055 [+0.007, +0.118]: 8x the episodes at equal training is no better, slightly
# worse; no train / held-out gap in either: own 0.595 / 0.540). The long arm (128 episodes, 36,000 steps) is running.
# Here: 1,024 episodes at 36,000 steps -- with it the 2 x 2 (128 | 1,024 episodes) x (12,000 | 36,000 steps) is whole.
#   rigid_pilot1024p_attn_direct_36k
# PREDICTION (written before this run and before the long arm ended): 0.49-0.54 on shard 8, at least 0.02 below
# what the long arm scores there (at 3x the training the 128 episodes are seen ~6.6 times a window and a gap opens;
# the 1,024 ~0.8 times). READ: 1,024 x 36k below 128 x 36k by >= 0.03 on held-out episodes = data becomes the lever
# once training is long enough, and the next rung is data x training together; level = neither size of data matters
# and what is left is the network or its inputs.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot1024p_attn_direct_36k
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[levers2] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 36000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[levers2] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${n}_shard1.log 2>&1 &
wait
echo "[levers2] $n scored $(date -u)"
