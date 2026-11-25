#!/bin/bash
# The mixed model's 9 clips (as m64R_render.sh) with the restoration: $1 = checkpoint, $2 = output dir, $3 = GPU.
# For a checkpoint trained without it, LSIM_RESTORE=1 turns it on at inference.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export CUDA_VISIBLE_DEVICES=$3
CLIPS="r1_test/shard_00000:4 r1_test/shard_00000:8 r1_test/shard_00000:0 r1_test/shard_00000:7 s1_test/shard_00000:9 s1_test/shard_00001:15 s1_test/shard_00000:22 s1_test/shard_00001:11 c1_test/shard_00000:0"
f=$1; O=$2
for c in $CLIPS; do
  sh=data/datasets/${c%%:*}; e=${c##*:}
  $L scripts/l2_render_cloth.py export $f $sh $e 7 end --hidden --out $O >> $O.log 2>&1
done
for d in $(ls -d $O/*/); do
  $B --background --python scripts/blender_render.py -- ${d%/} >> $O.log 2>&1
  $L scripts/l2_render_cloth.py compose ${d%/} --out $O >> $O.log 2>&1
done
$L scripts/l2_render_cloth.py sheet $(ls -d $O/*/ | sed 's:/$::') --out $O >> $O.log 2>&1
echo "[render] $f -> $O done $(date -u)"
