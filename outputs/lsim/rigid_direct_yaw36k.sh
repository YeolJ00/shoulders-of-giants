#!/bin/bash
# The working rigid recipe at the longer training (2026-11-24): all inputs given (--plain) + the random yaw and shift
# (--aug 1), which at 12,000 steps cost nothing on one frame (0.550 against 0.571, -0.015 [-0.041, +0.011]) and gave
# a steadier replay (never-moved bodies 0.5 / 0.8 / 3.4 mm against 0.9 / 2.1 / 7.8). Same 128 episodes and network,
# 36,000 steps, beside rigid_pilot128p_attn_direct_36k (no yaw: own 0.411, shard 1 0.501, shard 8 0.454):
#   rigid_pilot128p_attn_direct_yaw_36k
# PREDICTION (written before the run): level with the no-yaw run on unseen episodes (within 0.02 on both shards)
# with a smaller gap to its own episodes (own 0.43-0.47: every orientation is a new example, so it memorises its
# 128 episodes less), and resting bodies steadier in the replay. READ: level or better = the yaw goes into the
# scale-up recipe; worse by 0.03 or more on unseen episodes = at long training the yaw does cost, and the
# scale-up runs without it.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot128p_attn_direct_yaw_36k
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[yaw36k] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 36000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --aug 1 --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[yaw36k] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${n}_shard1.log 2>&1 &
wait
echo "[yaw36k] $n scored $(date -u)"
