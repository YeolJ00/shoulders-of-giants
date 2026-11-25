#!/bin/bash
# GPU 3: each 10 s-rollout fine-tune checkpoint (cloth_q64X) over 20 s on c1_long_test, and the same two towels rendered.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export CUDA_VISIBLE_DEVICES=3
for s in 50 100 150 199; do
  f=outputs/lsim/cloth_q64X/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_q64X.log; do sleep 120; done
  [ -f $f ] || break; sleep 30
  $L scripts/l2_cloth_long.py --ckpt $f --data c1_long_test --frames 590 --tag long20_q64X_step$s > outputs/lsim/long20_q64X_step$s.log 2>&1
  for e in 0 5; do
    $L scripts/l2_render_cloth.py export $f data/datasets/c1_long_test/shard_00000 $e 7 590 long20_q64X_step$s > /dev/null 2>&1
    d=$(ls -d outputs/lsim/render_cloth_model/shard_00000_e${e}_t7_long20_q64X_step$s | head -1)
    $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_q64X_blender.log 2>&1
    $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_q64X_blender.log 2>&1
  done
  echo "[x] step $s $(date -u)"
done
