#!/bin/bash
# The restoration at inference only (LSIM_RESTORE=1) on the 10 s-rollout cloth fine-tune (q64X step 199): 20 s on
# c1_long_test and the same two towels rendered, as q64X_eval.sh; GPU 5.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export LSIM_RESTORE=1 CUDA_VISIBLE_DEVICES=5
f=outputs/lsim/cloth_q64X/step199.pt
for e in 0 5; do
  $L scripts/l2_render_cloth.py export $f data/datasets/c1_long_test/shard_00000 $e 7 590 long20_q64X_step199_P > /dev/null 2>&1
  d=$(ls -d outputs/lsim/render_cloth_model/shard_00000_e${e}_t7_long20_q64X_step199_P | head -1)
  $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_q64X_P_blender.log 2>&1
  $L scripts/l2_render_cloth.py compose $d >> outputs/lsim/render_q64X_P_blender.log 2>&1
done
$L scripts/l2_cloth_long.py --ckpt $f --data c1_long_test --frames 590 --tag long20_q64X_step199_P > outputs/lsim/long20_q64X_step199_P.log 2>&1
echo "[cloth] done $(date -u)"
