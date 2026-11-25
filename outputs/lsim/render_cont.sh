#!/bin/bash
# The four fixed held-out cloth episodes with the continued local-frame network (cloth_q64_ownup_cont), tag localcont,
# into outputs/lsim/render_heldout_full beside origfull / localfull.   $1 = GPU (shared with a light job is fine)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_full
g=$1
for ep in "shard_00008 21" "shard_00008 17" "shard_00000 18" "shard_00008 50"; do
  set -- $ep
  D=$O/${1}_e${2}_t7_localcont
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/cloth_q64_ownup_cont/step23999.pt data/datasets/c1_test/$1 $2 7 end localcont --out $O
  CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $D --out $O
  echo "[rcont] $1 $2 ($?) $(date -u)"
done
echo "[rcont] done $(date -u)"
