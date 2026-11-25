#!/bin/bash
# Which hidden input costs the plain rigid transformer? (2026-11-24) rigid_direct_recipe.sh's split came out the
# other way round from its prediction: the random yaw and shift alone cost NOTHING (--plain --aug 1: 0.550 on the 32
# held-out episodes against --plain's 0.571, -0.015 [-0.041, +0.011], and a steadier replay), and the full recipe
# without them is as bad as with (0.747 against 0.765; --plain +0.163 [+0.080, +0.236]; resting bodies drift 4 / 16 /
# 107 mm). What is left differs from --plain in four things; the full recipe's loss ends at 1.0-1.8 where --plain's
# ends at 0.07-0.15 (loss = l_pos + l_par + l_f, one trunk, one gradient clip). One arm each, on top of --plain:
#   hid    --p_hidden 0.5                      parameter-slot groups hidden half the time (l_par on the trunk)
#   grip   --p_hide_grip 0.3                   the gripper's force slot hidden in 30% of windows (l_f on the trunk)
#   past   --all_past 0 --p_hist_noise 0.2     a random shorter past, flagged history noise
# PREDICTION (written before the runs; the last one on this failed, so held loosely): the hidden slots carry most
# of it (hid 0.68-0.76 on held-out episodes, with the resting drift), as on cloth, where the slot loss on the trunk
# cost the fit until --side_apart took it off; grip 0.58-0.66; past 0.56-0.62. READ: whichever arm is above 0.68
# with resting bodies drifting is the one to fix on the learning side (its loss off the trunk, or its weight)
# before claim 2 is asked of this network; none above 0.65 = they only cost together.
#   $1 = GPU (empty twice a minute apart), $2 = hid | grip | past
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; v=$2
case $v in
  hid)  A="--p_hidden 0.5";;
  grip) A="--p_hide_grip 0.3";;
  past) A="--all_past 0 --p_hist_noise 0.2";;
esac
n=rigid_pilot128p_attn_direct_$v
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[recipe2] $n GPU $g: $A $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain $A --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[recipe2] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
wait
echo "[recipe2] $n scored $(date -u)"
