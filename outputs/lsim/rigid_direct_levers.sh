#!/bin/bash
# The plain transformer on rigid objects: which lever next? (2026-11-23) rigid_pilot128p_attn_direct -- the attention
# trunk with NO physics layer (--direct), --plain with the noise and pushforward kept, 128 episodes, 12,000 steps,
# 1.6M parameters, 2.5 h on one GPU -- scores one frame 0.571 of constant velocity on 32 held-out episodes (own 32:
# 0.540), level with B1 (0.556) and ahead of B2 (0.631), both trained on all 50,048 episodes, and replays held-out
# episodes at their level or closer (median body that moved by 1 s: 4.5 mm; B1 6.5, B2 6.9, left in place 33.4).
# Predictions for that run (0.85-1.05; 8-20 mm) were wrong on the good side. The same network WITHOUT the noise
# learned nothing (0.982: constant velocity in every category), and the layer network with a learnable gripper force
# stayed at 1.500 (without it 1.491). No train / held-out gap at 128 episodes (0.03), so by the cloth track's
# reading the fit is limited by training, not by data. Two arms, same recipe, the user's order (fit, then data):
#   long   128 episodes, 36,000 steps (3x)             rigid_pilot128p_attn_direct_36k
#   data   1,024 episodes (8 shards), 12,000 steps     rigid_pilot1024p_attn_direct      (needs l2_overfit.py --episodes > 128)
# PREDICTION (written before the runs): long ends at 0.42-0.50 on its own episodes and 0.50-0.56 on held-out ones
# (a gap of 0.04-0.10 opens); data ends at 0.54-0.58 on held-out episodes with no gap (8x the episodes at equal
# steps buys < 0.03, as 10x did for cloth at equal compute). READ: long gains >= 0.03 on held-out = training length
# is still a lever at 128 episodes; long's own score falls and held-out does not = data is the lever, and the data
# arm should show it; neither moves held-out by 0.03 = the network's size or inputs.
#   $1 = GPU (empty twice a minute apart), $2 = long | data
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; v=$2
if [ "$v" = long ]; then n=rigid_pilot128p_attn_direct_36k; A="--episodes 128 --steps 36000"; else n=rigid_pilot1024p_attn_direct; A="--episodes 1024 --steps 12000"; fi
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[levers] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 $A --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[levers] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${n}_shard1.log 2>&1 &
wait
echo "[levers] $n scored $(date -u)"
