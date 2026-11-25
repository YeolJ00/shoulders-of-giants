#!/bin/bash
# Where the rigid layer model STARTS (2026-11-23). The clean-view pair (rigid_pilot128_clean.sh) ended above constant
# velocity too (held-out one frame: graph 1.620, attention 1.491; own 32: 1.449 / 1.235; the prediction "both below 1"
# failed, "attention level or ahead" held: -0.114 [-0.203, -0.031]). The body-frame breakdown on the same 32 held-out
# episodes (scratch rigid_where_many.py / rigid_where_join.py) says why: THE EXACT LAYER ALONE (force heads zeroed) IS
# AT 1.736 of constant velocity -- the networks moved it by 0.1-0.25 in 12,000 steps. Where a given gripper force acts
# (194 of 563 moving body-frames) the layer is 7.15 mm off, constant velocity 1.84, the networks 7.12 / 7.24 (they have
# no force on gripper edges), B1 / B2 1.82; where another object touches 6.5 (cv 2.4). Worst frames: a 31 g object
# pushed by 4.6 N against another object is thrown 150 mm in a frame (nothing balances the recorded force), a thin
# plate pushed low is turned 104 deg (the default's reaction passes through the centre of mass: the push's torque is
# unbalanced). B2 starts AT constant velocity by construction; the layer model starts 1.7x behind it.
# Two arms in the same clean view and budget (--plain --noise 0,0 --pf 0, 128 episodes, 12,000 steps), attention trunk:
#   rigid_pilot128c_attn_direct   --direct      no physics layer: the plain transformer (starts at constant velocity)
#   rigid_pilot128c_attn_grip     --grip_force  the layer, with a learned force on gripper edges too
# PREDICTION (written before the runs): direct ends BELOW 1 on held-out episodes (0.75-0.95; B2 on all 50k episodes
# is at 0.63) and on its own (0.65-0.90); --grip_force stays above 1 (1.20-1.45 held-out, from 1.49): it can now act
# on gripper-driven frames but must cancel forces many times the body's weight to a few per cent. READ: direct below
# 1 with the layer arms above it = at this data size the layer's starting point is the handicap, and the design
# question is the gripper force slot (integrated as an external force today; a reaction of the motion in the teacher)
# and the push's torque in the default. Direct NOT below 1 = 128 episodes x 12,000 steps teaches no design the
# one-frame map and the pilot size must go up before any design comparison. --grip_force below 1.2 = the network can
# learn the cancellation and the handle was what was missing.
#   $1 = GPU (empty twice a minute apart), $2 = name, $3.. = the arm's flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
S=/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad
g=$1; n=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[pilot128s] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb \
  --pf 0 --noise 0,0 --mix --plain "$@" --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[pilot128s] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L $S/rigid_where_many.py outputs/lsim/$n.pt 1 32 $S/where_${n#rigid_pilot128}_held.npz > $S/where_${n#rigid_pilot128}_held.log 2>&1 &
wait
echo "[pilot128s] $n scored $(date -u)"
