#!/bin/bash
# The cloth evaluations that decide a fine-tune: 96 rollouts against the baselines, the stretch, and the four
# held-out renders (as eval_cloth.sh, without G5 and G4).   outputs/lsim/eval_cloth_ft.sh <ckpt> <tag> <gpu>
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
CK=$1; TAG=$2; export CUDA_VISIBLE_DEVICES=$3; DATA=data/datasets/c1_test
echo "[eval] $TAG start $(date -u)"
$L scripts/l2_cloth_eval.py --ckpt $CK --data $DATA --shards 16 --n 96 --tag $TAG > outputs/lsim/eval_$TAG.log 2>&1
$L scripts/l2_cloth_strain.py --ckpt $CK --tag $TAG > outputs/lsim/strain_$TAG.log 2>&1
echo "[eval] $TAG scores done $(date -u)"
for spec in "$DATA/shard_00000 0 25" "$DATA/shard_00000 3 20" "$DATA/shard_00000 2 25" "$DATA/shard_00000 14 15"; do
  set -- $spec
  $L scripts/l2_render_cloth.py export $CK $1 $2 $3 60 $TAG >> outputs/lsim/render_$TAG.log 2>&1
  d=outputs/lsim/render_cloth_model/$(basename $1)_e$2_t$3_$TAG
  $B --background --python scripts/blender_render.py -- $d 640 480 48 >> outputs/lsim/render_$TAG.log 2>&1
  $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_$TAG.log 2>&1
done
echo "[eval] $TAG done $(date -u)"
