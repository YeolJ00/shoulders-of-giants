#!/bin/bash
# Windows that start from rest, in the main rigid recipe (2026-11-26). The release test (rigid_release.sh: the plain
# transformer on 1,024 v0r episodes, 12,000 steps, --p_start 0 / 0.1) FAILED its question -- the real cube still does
# not slide (81.2 against 82.2 mm; predicted under 20), and on v0r's own unseen ramp objects (shard 9, scratch
# ramp_release_check.py) neither network starts a slide from rest (3-5 mm in 1.2 s against the teacher's 320;
# from frame 8 with true history the start arm follows 60%, the control 37%). But it found something else: the
# windows from frame 0 lowered one frame on unseen episodes, v0 shard 8 0.787 -> 0.729 (-0.057 [-0.092, -0.024], 27
# of 32) and v0r shard 9 0.895 -> 0.816 (-0.079 [-0.114, -0.048], 28 of 32). v0 episodes start with objects lifted
# and let go at frame 0, so these windows are the drops' first frames, never drawn before (starts were uniform from
# frame 1). One run each (the RNG streams differ after the first draw), so seed noise is not excluded. Here in the
# main recipe at the 36,000-step budget, against rigid_pilot1024p_attn_direct_36k (same everything, p 0):
#   rigid_pilot1024p_attn_direct_36k_start   --episodes 1024 --steps 36000 --p_start 0.1   (~7.3 h, one GPU)
# scored on its own episodes (shard 0), shard 8 and shard 64 (32 episodes each), and on the real ramp.
# PREDICTION (written before the run): one frame lower by 0.03-0.08 on shards 8 and 64 (reference 0.447 / 0.504),
# own lower by as much (no gap opens), replays of moved bodies level or closer; real ramp still ~80 mm.
# READ: lower by 0.03 or more on both unseen shards = --p_start 0.1 joins the rigid recipe (the 8,192 and long runs
# take it); level (within 0.02) = the 12k v0r gain was seed noise or v0r's ramps, and the recipe stays.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot1024p_attn_direct_36k_start
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[pstart] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 36000 --pair_norm --coulomb \
  --pf 6 --noise 0.003,0.015 --mix --plain --trunk attn --d 128 --layers 6 --direct --p_start 0.1 \
  --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[pstart] $n trained ($?) $(date -u)"
for x in "own 0" "held 8" "held64 64"; do set -- $x
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard $2 --tag $1 \
    > outputs/lsim/small_fit_${n}_$1.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_phys101.py --ckpt outputs/lsim/$n.pt --out docs/results/l2_phys101_$n.json \
  > outputs/lsim/phys101_$n.log 2>&1 &
wait
echo "[pstart] $n scored $(date -u)"
