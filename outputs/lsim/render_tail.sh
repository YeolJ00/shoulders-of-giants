#!/bin/bash
# The blow-up tail of the third cloth stretch, looked at (2026-11-25). cont_tail_check.sh: the 72k network ends above 2x
# area on 27 / 21 of 336 held-out rollouts at its step 20,000 / 23,999, the 48k network on 12 / 6 -- the tail is real.
# The four fixed episodes do not show it (72k level or better, push no longer stretched 4x). Render two held-out
# episodes where the 72k network ends at 15-17x area and the 48k one keeps the cloth (lift-and-place shard 0 ep 14:
# 0.74x vs 16.7x; fold shard 3 ep 36: 0.88x vs 14.7x), both networks, to see what the failure is: an explosion from one
# frame (a bad step), or a slow stretch that compounds.
# PREDICTION: an explosion that starts when the cloth contacts something (a pile, the table after a drop), not a slow
# stretch from the start. Evaluation only, one GPU.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_tail
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[tailr] GPU $g $(date -u)"
mkdir -p $O
for ep in "shard_00000 14" "shard_00003 36"; do
  set -- $ep
  for x in "cloth_q64_ownup_cont localcont" "cloth_q64_ownup_cont2 localcont2"; do
    set -- $ep $x
    D=$O/${1}_e${2}_t7_$4
    ( CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$3/step23999.pt data/datasets/c1_test/$1 $2 7 end $4 --out $O
      CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
      $L scripts/l2_render_cloth.py compose $D --out $O
      echo "[tailr] render $1 $2 $4 ($?) $(date -u)" ) &
  done
done
wait
echo "[tailr] done $(date -u)"
