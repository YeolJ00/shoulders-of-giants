#!/bin/bash
# Which part of the full recipe costs the plain rigid transformer? (2026-11-24) rigid_pilot128_attn_direct -- the
# recipe AS IT IS (random yaw and shift per window, hidden parameter slots, hidden gripper, a random shorter past,
# history noise) -- ended at 0.765 on the 32 held-out episodes (own 0.741) where the --plain arm (none of those) is at
# 0.571: +0.175 [+0.099, +0.243], worse in 28 of 32, and its replay lets resting bodies drift (never-moved bodies 7 /
# 22 / 142 mm at 1 s / 2 s / end against 0.9 / 2.1 / 7.8). The prediction (0.60-0.72) failed. Two arms split it, same
# 128 episodes, steps, network and noise:
#   yaw     --plain --aug 1   only the random yaw and shift added to the plain view   rigid_pilot128p_attn_direct_yaw
#   noyaw   --aug 0           the full recipe without the yaw and shift               rigid_pilot128_attn_direct_noyaw
# PREDICTION (written before the runs): the yaw carries most of it -- yaw 0.70-0.78 on held-out episodes, noyaw
# 0.58-0.64 -- as on cloth, where a world-frame network had to learn every orientation separately (one episode: 0.10
# without the yaw, 0.64 with). READ: yaw >= 0.70 and noyaw <= 0.64 = a yaw-invariant input is the next thing for
# rigid (the cloth network's local frame was worth -0.10 at full data); the reverse = the hidden slots or gripper
# cost it, and the slot machinery has to be looked at before claim 2 can be asked of this network.
#   $1 = GPU (empty twice a minute apart), $2 = yaw | noyaw
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; v=$2
if [ "$v" = yaw ]; then n=rigid_pilot128p_attn_direct_yaw; A="--plain --aug 1"; else n=rigid_pilot128_attn_direct_noyaw; A="--aug 0"; fi
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[recipe] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  $A --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[recipe] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
wait
echo "[recipe] $n scored $(date -u)"
