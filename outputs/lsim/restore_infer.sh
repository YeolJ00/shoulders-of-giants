#!/bin/bash
# The restoration at inference only (LSIM_RESTORE=1) on existing checkpoints: the cloth model over 20 s (GPU 5) and the
# mixed model's 9 clips (GPU 6), as long20_q64X_step199 and m64R_render.sh / render_mixed were run.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export LSIM_RESTORE=1
(
export CUDA_VISIBLE_DEVICES=5
f=outputs/lsim/cloth_q64X/step199.pt
$L scripts/l2_cloth_long.py --ckpt $f --data c1_long_test --frames 590 --tag long20_q64X_step199_P > outputs/lsim/long20_q64X_step199_P.log 2>&1
for e in 0 5; do
  $L scripts/l2_render_cloth.py export $f data/datasets/c1_long_test/shard_00000 $e 7 590 long20_q64X_step199_P > /dev/null 2>&1
  d=$(ls -d outputs/lsim/render_cloth_model/shard_00000_e${e}_t7_long20_q64X_step199_P | head -1)
  $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_q64X_P_blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_q64X_P_blender.log 2>&1
done
echo "[cloth] done $(date -u)"
) &
(
export CUDA_VISIBLE_DEVICES=6
CLIPS="r1_test/shard_00000:4 r1_test/shard_00000:8 r1_test/shard_00000:0 r1_test/shard_00000:7 s1_test/shard_00000:9 s1_test/shard_00001:15 s1_test/shard_00000:22 s1_test/shard_00001:11 c1_test/shard_00000:0"
f=outputs/lsim/cloth_m64/step23999.pt
O=outputs/lsim/render_mixed_P0
for c in $CLIPS; do
  sh=data/datasets/${c%%:*}; e=${c##*:}
  $L scripts/l2_render_cloth.py export $f $sh $e 7 end --hidden --out $O >> $O.log 2>&1
done
for d in $(ls -d $O/*/); do
  $B --background --python scripts/blender_render.py -- ${d%/} >> $O.log 2>&1
  $L scripts/l2_render_cloth.py compose ${d%/} --out $O >> $O.log 2>&1
done
$L scripts/l2_render_cloth.py sheet $(ls -d $O/*/ | sed 's:/$::') --out $O >> $O.log 2>&1
echo "[mixed] done $(date -u)"
) &
wait
