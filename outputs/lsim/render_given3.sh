#!/bin/bash
# The every-input-given third stretch, looked at (2026-11-25). cloth_q64_ownup_cont2_given (a third 24,000 steps from
# cloth_q64_ownup_cont/step23999.pt with --given) against cloth_q64_ownup_cont2 (the same stretch, hidden-input
# recipe), both rolled out with the material given on the same 336 held-out episodes: blow-ups (area > 2x at the
# end) 5 against 28, strain mean 4.2% against 24.3%, one frame 0.475 against 0.486, positions level. Render both
# networks, material given, from frame 7 to the end, on three episodes the old network blows up and the new one
# keeps (fold shard 2 ep 27: 17x -> 1.0x, end 162 -> 3 mm; lift-drop shard 2 ep 54: 19x -> 1.0x, 144 -> 36 mm;
# shake shard 12 ep 38: 19x -> 0.9x, 276 -> 153 mm) and one of the new network's five blow-ups (fold shard 13 ep 61:
# 15x old, 4.4x new).
# PREDICTION: on the three kept episodes the new network's cloth keeps its size and shape throughout (no bunching
# into a ball), and the old network's bunches up at the landing or fold as in the pushforward renders; on the fold
# failure both bunch, the new one later or smaller. Evaluation only, one GPU.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_given
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rg] GPU $g $(date -u)"
mkdir -p $O
for ep in "shard_00002 27" "shard_00002 54" "shard_00012 38" "shard_00013 61"; do
  for x in "cloth_q64_ownup_cont2 old" "cloth_q64_ownup_cont2_given given"; do
    set -- $ep $x
    D=$O/${1}_e${2}_t7_$4
    ( CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$3/step23999.pt data/datasets/c1_test/$1 $2 7 end $4 --out $O
      CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
      $L scripts/l2_render_cloth.py compose $D --out $O
      echo "[rg] render $1 $2 $4 ($?) $(date -u)" ) &
  done
done
wait
echo "[rg] done $(date -u)"
