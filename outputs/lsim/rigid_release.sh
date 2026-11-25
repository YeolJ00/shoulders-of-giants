#!/bin/bash
# Release from rest, for the real ramp (2026-11-25). The plain rigid transformer (no physics layer, every input given)
# does not slide Physics 101's cube down the real 20 deg board: 82 mm trained on v0, 81.2 on 128 episodes of the ramp
# supplement v0r (rigid_direct_ramp.sh; the Coulomb default alone is within 1 mm). That run's stated reason --
# "teacher episodes are recorded after the scene has settled, so rest on a slope always stays rest in the data" -- is
# wrong: v0r puts its ramp objects at rest on the plank at frame 0, after the settle (lsim.data.episode.run_batch,
# step 3), and of the 288 ramp objects in its first 1,024 episodes 196 then slide (177 by over 5 cm within 8 frames;
# scratch ramp_release_count.py). What training never draws is the window the real test is: l2_phys101.py rolls the
# cube out from 8 identical frames of rest ("released at now"), while l2_overfit.py draws window starts from frame 1,
# so no training window has a whole past of rest. New: l2_overfit.py --p_start p (a window starts at frame 0 with
# probability p; its past is frame 0 repeated -- the scene held still, then let go). Same recipe as
# rigid_direct_ramp.sh, 1,024 episodes of v0r (shards 0-7), 12,000 steps, both on one GPU (1.6M-parameter runs):
#   rigid_ramp1024_ctl     --p_start 0   (the control: the ramp data as drawn before)
#   rigid_ramp1024_start   --p_start 0.1 (one window in ten starts from rest; ~23% of those hold a ramp object)
# scored by l2_phys101.py (93 real slides, friction swept), and one frame / replay on 32 held-out episodes of v0r
# (shard 9) and of v0 (shard 8).
# PREDICTION (written before the run): the control does not slide (70-85 mm). The start arm slides: median RMS under
# 20 mm (the slide law is a function of slope and the given friction, shown in ~200 releases). One frame on both
# held-out sets within 0.03 of the control.
# READ: under 20 mm = the real ramp needs no physics layer, only training windows that start from rest -- the layer's
# last measured case goes, and release windows join the rigid recipe; 20-70 = partly, more releases (episodes or p)
# next; control level = the network does not learn the slide law from rest even when shown ~200 releases, and the
# default stays the only route to the real ramp (the user's decision).
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rel] GPU $g $(date -u)"
R="--data data/datasets/v0r --episodes 1024 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix --plain --trunk attn --d 128 --layers 6 --direct"
# smoke: every window from rest, 30 steps
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py $R --episodes 16 --steps 30 --p_start 1.0 --out outputs/lsim/tmp/rigid_release_smoke.pt \
  > outputs/lsim/rigid_release_smoke.log 2>&1 || { echo "[rel] smoke FAILED $(date -u)"; exit 1; }
rm -f outputs/lsim/tmp/rigid_release_smoke.pt
echo "[rel] smoke ok $(date -u)"
run() {   # name p_start
  local n=$1
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py $R --steps 12000 --p_start $2 --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
  echo "[rel] $n trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_phys101.py --ckpt outputs/lsim/$n.pt --out docs/results/l2_phys101_$n.json > outputs/lsim/phys101_$n.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --data data/datasets/v0r --episodes 32 --shard 9 --tag v0r_held > outputs/lsim/small_fit_${n}_v0r_held.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
  wait
  echo "[rel] $n scored $(date -u)"
}
run rigid_ramp1024_ctl 0 &
run rigid_ramp1024_start 0.1 &
wait
echo "[rel] done $(date -u)"
