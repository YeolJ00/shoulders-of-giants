#!/bin/bash
# GPU 2: the mixed fine-tune's checkpoints (cloth_m64R: edge 10 + 30-frame rollouts) rendered on the same 9 clips as render_mixed.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export CUDA_VISIBLE_DEVICES=2
CLIPS="r1_test/shard_00000:4 r1_test/shard_00000:8 r1_test/shard_00000:0 r1_test/shard_00000:7 s1_test/shard_00000:9 s1_test/shard_00001:15 s1_test/shard_00000:22 s1_test/shard_00001:11 c1_test/shard_00000:0"
for s in 500 1000 1999; do
  f=outputs/lsim/cloth_m64R/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_m64R.log; do sleep 120; done
  [ -f $f ] || break; sleep 30
  O=outputs/lsim/render_mixed_R$s
  for c in $CLIPS; do
    sh=data/datasets/${c%%:*}; e=${c##*:}
    $L scripts/l2_render_cloth.py export $f $sh $e 7 end --hidden --out $O >> $O.log 2>&1
  done
  for d in $(ls -d $O/*/); do
    $B --background --python scripts/blender_render.py -- ${d%/} >> $O.log 2>&1
    $L scripts/l2_render_cloth.py compose ${d%/} --out $O >> $O.log 2>&1
  done
  $L scripts/l2_render_cloth.py sheet $(ls -d $O/*/ | sed 's:/$::') --out $O >> $O.log 2>&1
  echo "[R] step $s rendered $(date -u)"
done
