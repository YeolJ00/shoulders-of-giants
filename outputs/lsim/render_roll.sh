#!/bin/bash
# The long-rollout trial's 3.3 s arm, looked at (2026-11-26). cloth_roll100 (400 steps from the hidden-input third
# stretch with a loss on its own 100-frame runs against the teacher's) against cloth_roll_ctl (the same 400 steps
# without it): blow-ups 14 against 30 of 336, closer by 4.5 / 6.2 mm at 2 s / 3.8 s, the typical cloth slightly more
# stretched (median 1.8% against 1.0%). Both networks, material given as scored, from frame 7 to the end, on two
# episodes the control blows up and the rollout arm keeps: fold shard 14 ep 19 (21x -> 0.97x, end 120 -> 45 mm) and
# shake shard 3 ep 28 (16x -> 0.91x, 104 -> 50 mm; also a pushforward-trial blow-up).
# PREDICTION: the control's cloth bunches into a ball after the fold / the shake's release, as in the earlier
# renders; the rollout arm's keeps its size throughout, with a visibly even, slight stretch rather than none.
# Evaluation only, one GPU.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_roll
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rr] GPU $g $(date -u)"
mkdir -p $O
for ep in "shard_00014 19" "shard_00003 28"; do
  for x in "cloth_roll_ctl ctl" "cloth_roll100 roll"; do
    set -- $ep $x
    D=$O/${1}_e${2}_t7_$4
    ( CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$3/step399.pt data/datasets/c1_test/$1 $2 7 end $4 --out $O
      CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
      $L scripts/l2_render_cloth.py compose $D --out $O
      echo "[rr] render $1 $2 $4 ($?) $(date -u)" ) &
  done
done
wait
echo "[rr] done $(date -u)"
