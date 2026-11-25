#!/bin/bash
# The four fixed held-out cloth episodes with the every-input-given network on 2,000 episodes (cloth_cov_seven_ownup_given,
# tag local2kgiven; material given, as it trained), into outputs/lsim/render_heldout_pair beside local2k (the usual
# hidden-input recipe, same data and steps). Evaluation only.   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_pair
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
for ep in "shard_00008 21" "shard_00008 17" "shard_00000 18" "shard_00008 50"; do
  set -- $ep
  D=$O/${1}_e${2}_t7_local2kgiven
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/cloth_cov_seven_ownup_given/step11999.pt data/datasets/c1_test/$1 $2 7 end local2kgiven --out $O
  CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $D --out $O
  echo "[rgiven] $1 $2 ($?) $(date -u)"
done
echo "[rgiven] done $(date -u)"
