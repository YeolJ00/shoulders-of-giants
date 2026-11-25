#!/bin/bash
# GPU 3: the cloth models over 20 s on c1_long_test (alone from frame 7), two rendered; then rq509's PGND evals.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export CUDA_VISIBLE_DEVICES=3
for c in still layer outputs/lsim/cloth_q64L/step1500.pt outputs/lsim/cloth_q64/step23999.pt; do
  t=$(basename $(dirname $c))_$(basename $c .pt); [ $c = still ] || [ $c = layer ] && t=$c
  $L scripts/l2_cloth_long.py --ckpt $c --data c1_long_test --frames 590 --tag long20_$t > outputs/lsim/long20_$t.log 2>&1
  echo "[g3] 20 s $t $(date -u)"
done
for e in 0 5; do
  $L scripts/l2_render_cloth.py export outputs/lsim/cloth_q64L/step1500.pt data/datasets/c1_long_test/shard_00000 $e 7 590 long20 > outputs/lsim/render_long20_$e.log 2>&1
done
for d in outputs/lsim/render_cloth_model/*long20*; do
  $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_long20_blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_long20_blender.log 2>&1
done
echo "[g3] renders $(date -u)"
for s in 250 500 750 1000; do
  f=outputs/lsim/cloth_rq509/step$s.pt
  until [ -f $f ] || ! ps -p 1636210 > /dev/null; do sleep 120; done
  [ -f $f ] || break; sleep 30
  $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag rq509_step$s > outputs/lsim/pgnd_eval_rq509_step$s.log 2>&1
  echo "[g3] rq509 step $s evaluated $(date -u)"
done
