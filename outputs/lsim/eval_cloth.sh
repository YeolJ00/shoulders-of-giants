#!/bin/bash
# Every cloth evaluation for one checkpoint: rollouts against the baselines, G5 ensembles, G4 slot inference, renders.
#   outputs/lsim/eval_cloth.sh <ckpt> <tag> <data dir> <skip shards> <gpu>
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
CK=$1; TAG=$2; DATA=$3; SKIP=$4; GPU=${5:-4}
export CUDA_VISIBLE_DEVICES=$GPU
echo "[eval] start $(date -u)"
$L scripts/l2_cloth_eval.py --ckpt $CK --data $DATA --skip $SKIP --shards 16 --n 96 --tag $TAG > outputs/lsim/eval_$TAG.log 2>&1
$L scripts/l2_cloth_eval.py --ckpt $CK --data $DATA --skip $SKIP --shards 16 --n 24 --ensemble 8 --chaos unc --tag ${TAG}_g5 > outputs/lsim/eval_${TAG}_g5.log 2>&1
$L scripts/l2_cloth_slots.py --ckpt $CK --data $DATA --skip $SKIP --shards 16 --n 256 --tag $TAG > outputs/lsim/slots_$TAG.log 2>&1
echo "[eval] scores done $(date -u)"
# held-out c1_test episodes, one per kind of motion: a fold (shard 0 e0), a drag past two objects (e3), a crumpled
# towel lifted and dropped (e2), a held towel laid down (e14); "shard episode t0"
for spec in "$DATA/shard_00000 0 25" "$DATA/shard_00000 3 20" "$DATA/shard_00000 2 25" "$DATA/shard_00000 14 15"; do
  set -- $spec
  $L scripts/l2_render_cloth.py export $CK $1 $2 $3 60 $TAG >> outputs/lsim/render_$TAG.log 2>&1
  d=outputs/lsim/render_cloth_model/$(basename $1)_e$2_t$3_$TAG
  $B --background --python scripts/blender_render.py -- $d 640 480 48 >> outputs/lsim/render_$TAG.log 2>&1
  $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_$TAG.log 2>&1
done
echo "[eval] done $(date -u)"
