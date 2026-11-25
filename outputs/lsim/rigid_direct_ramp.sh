#!/bin/bash
# Does ramp data let the plain rigid transformer slide on the real board? (2026-11-24) The layer's remaining case:
# Physics 101's cardboard cubes on a 20 deg board -- the Coulomb default alone 0.99 mm, full v1 26.7, B1 82 and B2 81
# (they never slide, with v0r's ramp episodes in their training), and the plain transformer trained on v0's first
# shard 82.1 (that shard has no ramp at all). Here the same network, recipe and steps on the 128 episodes of v0r's
# first shard (the ramp supplement), then l2_phys101.py:
#   rigid_pilot128p_attn_direct_ramp
# PREDICTION (written before the run): it still does not slide -- median RMS 70-85 mm. Reason: teacher episodes are
# recorded after the scene has settled, so a body with eight frames of rest on a slope is, in the data, always one
# that stays (if it could slide it already had); the real cube is released at t = 0 from a history of rest, a case
# the data cannot contain and the Coulomb default gets from the law. READ: under 10 mm = ramp data is enough and the
# layer's last case goes; 70+ mm = a network without the layer needs RELEASE events in its data (an object held on
# a slope, then let go: the identifiability rule of PHASE1_AMENDMENT.md applied to static friction), or the default.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot128p_attn_direct_ramp
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[ramp] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0r --episodes 128 --steps 12000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix \
  --plain --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[ramp] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_phys101.py --ckpt outputs/lsim/$n.pt --out docs/results/l2_phys101_pilot128p_attn_direct_ramp.json > outputs/lsim/phys101_$n.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --data data/datasets/v0r --episodes 32 --shard 1 --tag v0r_held > outputs/lsim/small_fit_${n}_v0r_held.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
wait
echo "[ramp] $n scored $(date -u)"
