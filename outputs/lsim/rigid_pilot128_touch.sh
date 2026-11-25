#!/bin/bash
# The switch (2026-11-23): Simulator(touch_cv=True) -- a rigid body with the gripper or another moving body within
# reach gets no default in that frame (no support, gravity and the given gripper force cancelled), so a zero output
# is constant velocity there; a body in flight or alone on fixed surfaces keeps the exact layer. Why: the layer alone
# is at 1.74 of constant velocity on the 32 held-out episodes' moving body-frames and the fully trained v1 at 1.80
# (scripts/l2_rigid_where.py; DIRECTION.md "The rigid layer model starts behind constant velocity"); the
# parameter-free switch is at 0.90 (checked on the built option with the heads zeroed: 0.895 on 12 episodes against
# 0.891 from the saved columns; at rest 0.02 mm against the layer's 0.20). Same 128 episodes, 12,000 steps,
# attention trunk as rigid_pilot128_start.sh's arms:
#   clean   --plain --noise 0,0 --pf 0   rigid_pilot128c_attn_touch   beside rigid_pilot128c_attn_direct / _attn (1.491)
#   noise   --plain, noise and pushforward kept   rigid_pilot128p_attn_touch   beside rigid_pilot128p_attn_direct
# PREDICTION (written before the runs, and before the direct arms' results): clean ends at 0.70-0.85 on held-out
# episodes (own 0.60-0.80) and BELOW the direct arm by 0.03-0.15 paired over episodes (it starts 0.10 ahead: flight
# and sliding alone on a surface are exact); with the noise 0.85-1.05 on one frame, and its replay has the median
# body that moved by 1 s closer than the direct arm's. READ: not below the direct arm on one frame = the layer's
# domain buys nothing for teacher fit at this size, and its case rests on the real ramp alone; above 1 = the switch's
# frame-to-frame change of default costs more than its starting point gives.
#   $1 = GPU (empty twice a minute apart), $2 = clean | noise
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; v=$2
if [ "$v" = clean ]; then n=rigid_pilot128c_attn_touch; R="--pf 0 --noise 0,0"; else n=rigid_pilot128p_attn_touch; R="--pf 6 --noise 0.003,0.015"; fi
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[pilot128t] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb \
  $R --mix --plain --trunk attn --d 128 --layers 6 --touch_cv --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[pilot128t] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_rigid_where.py score --ckpt outputs/lsim/$n.pt --shard 1 --tag ${n#rigid_pilot128}_held > outputs/lsim/rigid_where_${n}.log 2>&1 &
wait
echo "[pilot128t] $n scored $(date -u)"
