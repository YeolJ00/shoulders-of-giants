#!/bin/bash
# The rigid small-fit test with whole-scene attention in place of the graph (2026-11-23). rigid_fit1.sh's two arms
# exactly (one v0 episode, v1's recipe --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix, 12,000 steps by
# scripts/l2_overfit.py) with --trunk attn --d 128 --layers 6 (1.6M parameters; the graph is 2.4M):
#   rigid_fit_1_attn         the trainer's view as it is       graph: one frame 0.597 of constant velocity
#   rigid_fit_1_attn_plain   --plain                            graph: 0.358; replay median body 3 / 4.5 / 133 mm
# Evidence for it: four fixed windows 0.000 / 0.000 (graph 0.335 / 0.134); every moving frame of the episode, one-step
# windows without noise, 4,000 steps: 0.160 (graph 0.214), both still falling.
# PREDICTION (written before the run): --plain ends 0.20-0.28 on one frame (below the graph's 0.358: the reach and the
# per-vertex force are no longer the floor, training length is), its replay at 1 s / 2 s no worse than the graph's
# 3 / 4.5 mm median body and the end still above 50 mm (the late poke is an EMPTY slot: generated, not given). As it
# is: 0.40-0.50 (the graph lost 0.16-0.24 to the random yaw; rotary positions on world coordinates are not
# yaw-invariant either, so the same loss is expected). Wrong if --plain stays above 0.30 (then one-step fit on the
# chunk windows is limited by something the four windows did not contain) or the as-it-is arm matches --plain within
# 0.05 (then the yaw is not what costs this trunk).
#   $1 = GPU (empty twice a minute apart), $2 = name, $3.. = the arm's flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
S=/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad
g=$1; n=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rigid1a] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1 --steps 12000 --pair_norm --coulomb \
  --pf 6 --noise 0.003,0.015 --mix --trunk attn --d 128 --layers 6 "$@" --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[rigid1a] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 1 > outputs/lsim/small_fit_$n.log 2>&1
CUDA_VISIBLE_DEVICES=$g $L $S/rigid_where.py outputs/lsim/$n.pt > outputs/lsim/${n}_where.log 2>&1
echo "[rigid1a] $n scored ($?) $(date -u)"
