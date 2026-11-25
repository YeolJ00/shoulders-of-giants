#!/bin/bash
# After the third cloth segment (2026-11-25; cloth_q64_ownup_cont2, 72k steps: one frame 0.486 against 0.520, rollouts
# 14.5 / 53.5 / 77.9 against 15.0 / 57.2 / 85.1 mm, but strain mean 16.6% against 6.1% and area mean 1.58 against
# 1.04 with the median better -- a tail of cloths blowing up appeared). Evaluation only, one GPU, packed:
#  (1) PGND real cloth before any real data (tag q64_ownup_cont2), against the 48k network's 25.6 / 34.7 mm.
#      PREDICTION: 22-25 mm at 2 s, 30-36 at 4 s (the 24k -> 48k step gave -5.6 / -10.2; half a doubling here).
#  (2) the four fixed held-out episodes rendered (tag localcont2, render_heldout_full beside localcont): does the tail
#      show on them?
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_full
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[c2x] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_q64_ownup_cont2/step23999.pt \
  --tag q64_ownup_cont2 > outputs/lsim/pgnd_eval_q64_ownup_cont2.log 2>&1 &
for ep in "shard_00008 21" "shard_00008 17" "shard_00000 18" "shard_00008 50"; do
  set -- $ep
  D=$O/${1}_e${2}_t7_localcont2
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/cloth_q64_ownup_cont2/step23999.pt data/datasets/c1_test/$1 $2 7 end localcont2 --out $O
  CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $D --out $O
  echo "[c2x] render $1 $2 ($?) $(date -u)"
done
wait
echo "[c2x] done $(date -u)"
