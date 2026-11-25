#!/bin/bash
# The next rung for the plain rigid transformer, all inputs given (2026-11-24). The 2 x 2 is whole (one frame /
# constant velocity on the 32 unseen episodes of shard 8; all bodies' replay median at 2 s):
#                         12,000 steps      36,000 steps
#     128 episodes        0.539 / 5.6 mm    0.454 / 4.8 mm      (own episodes 0.540 -> 0.411: a gap opens)
#   1,024 episodes        0.568 / 4.9 mm    0.447 / 2.3 mm      (own 0.595 -> 0.476: no gap)
# One frame is set by the number of steps, almost whatever the data (1,024 against 128 at 36,000 steps: -0.012
# [-0.045, +0.019]); more episodes close the train / unseen gap and make the replays closer (bodies that moved: 2.9
# against 5.8 mm median at 2 s, closer in 76%; against B1 -6.1 [-8.2, -3.8] mm at 2 s and -0.133 [-0.234, -0.045] on
# one frame: the first rigid network clearly ahead of the best baseline on both). The prediction for the fourth cell
# (0.49-0.54, at least 0.02 below the 128-episode run) missed twice: lower than predicted, and level with it.
# Here the same 1,024 episodes for three times the steps again:
#   rigid_pilot1024p_attn_direct_108k     --episodes 1024 --steps 108000 (~22 h on one GPU)
# PREDICTION (written before the run): one frame on shard 8 0.37-0.42 (12,000 -> 36,000 gave -0.12; expect -0.03
# to -0.08 more), still no train / unseen gap (own within 0.03 of unseen), replay of the bodies that moved 2.0-2.9
# mm median at 2 s. READ: a gain of 0.03 or more on unseen episodes = training length is still the lever and the
# scale-up is a long run; under 0.02 = this 1.6M network has reached its level under this recipe and the next
# thing to vary is the network (size with long training, or its inputs).
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot1024p_attn_direct_108k
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[long2] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 108000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[long2] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${n}_shard1.log 2>&1 &
wait
echo "[long2] $n scored $(date -u)"
