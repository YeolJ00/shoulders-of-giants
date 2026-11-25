#!/bin/bash
# Each restoration fine-tune checkpoint as it appears: cloth (q64XP, q64XS) over 20 s on c1_long_test + the two towels
# rendered (GPU 5); mixed (m64P) shape over held-out episodes + the 9 clips rendered (GPU 6).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
cloth() {   # run step
  f=outputs/lsim/cloth_$1/step$2.pt; t=long20_$1_step$2
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_$1.log; do sleep 120; done
  [ -f $f ] || return; sleep 30
  for e in 0 5; do
    CUDA_VISIBLE_DEVICES=5 $L scripts/l2_render_cloth.py export $f data/datasets/c1_long_test/shard_00000 $e 7 590 $t > /dev/null 2>&1
    d=outputs/lsim/render_cloth_model/shard_00000_e${e}_t7_$t
    CUDA_VISIBLE_DEVICES=5 $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_restore_blender.log 2>&1
    $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_restore_blender.log 2>&1
  done
  CUDA_VISIBLE_DEVICES=5 $L scripts/l2_cloth_long.py --ckpt $f --data c1_long_test --frames 590 --tag $t > outputs/lsim/$t.log 2>&1
  echo "[cloth] $1 step $2 $(date -u)"
}
mixed() {   # step
  f=outputs/lsim/cloth_m64P/step$1.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_m64P.log; do sleep 120; done
  [ -f $f ] || return; sleep 30
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_shape_long.py --ckpt $f --tag m64P_step$1 > outputs/lsim/shape_long_m64P_step$1.log 2>&1
  bash outputs/lsim/render_mixed_P.sh $f outputs/lsim/render_mixed_P$1 6
  echo "[mixed] step $1 $(date -u)"
}
( for s in 50 100 150 199; do cloth q64XP $s; cloth q64XS $s; done ) &
( for s in 500 1000 1500 1999; do mixed $s; done ) &
wait
