#!/bin/bash
# Beyond memorising, rigid (2026-11-23): is whole-scene attention a better LEARNER than the graph, or only a better
# memoriser? On one episode it fits every moving frame to 0.033 of constant velocity where the graph stalls at 0.182
# (12,000 steps on 68 one-step windows, ~1,400 views each). Here: the 128 episodes of v0's first shard (22k windows),
# v1's recipe as it is (--pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix; random yaw, hidden slots), 12,000
# steps x batch 4 = 48k window views (~2 a window: nothing can be memorised), scored by l2_small_fit_rigid.py on 32 of
# its own episodes and on the first 32 of the second shard (never trained on):
#   rigid_pilot128_graph   the network as built (2.4M)
#   rigid_pilot128_attn    --trunk attn --d 128 --layers 6 (1.6M)
# PREDICTION (written before the run): the attention trunk is ahead on one frame on HELD-OUT episodes by 0.05 or more
# (paired over the 32 episodes, interval clear of zero), because part of the graph's error is structural at any data
# size (a body landing or hit from beyond the 3 cm contact radius within a frame has no edge; v0 bodies travel up to
# 6 cm a frame) -- and both are far from fitted (0.6-0.9), with no train / held-out gap at 2 views a window. Wrong if
# the two are level on held-out episodes (then attention only memorises better) or the graph is ahead (then its
# contact edges are the better prior at small data, and the reach should be given to it another way).
#   $1 = GPU (empty twice a minute apart), $2 = name, $3.. = the arm's flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[pilot128] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb \
  --pf 6 --noise 0.003,0.015 --mix "$@" --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[pilot128] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
wait
echo "[pilot128] $n scored $(date -u)"
