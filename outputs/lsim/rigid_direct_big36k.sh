#!/bin/bash
# Size with long training, rigid (2026-11-24). The 8.5M plain transformer (d256 x 8) was no better than the 1.6M one
# at 12,000 steps on 128 episodes (0.617 against 0.571 held-out), but larger networks often need more steps; the 2 x 2
# says steps set the one-frame fit. Here the 8.5M network on 1,024 episodes for 36,000 steps, against the 1.6M
# network's 0.447 on shard 8 (rigid_pilot1024p_attn_direct_36k), all inputs given, same recipe:
#   rigid_pilot1024p_attn_direct_big36k
# PREDICTION (written before the run): within 0.04 of the 1.6M network on shard 8 (0.41-0.48), replay level. READ:
# 0.03 or more better = capacity matters once trained long enough (scale the network with the steps); level or
# worse = size is not the lever at this data and recipe, and the 108,000-step run says how far steps alone go.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot1024p_attn_direct_big36k
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[big36k] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 36000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --trunk attn --d 256 --layers 8 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[big36k] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${n}_shard1.log 2>&1 &
wait
echo "[big36k] $n scored $(date -u)"
