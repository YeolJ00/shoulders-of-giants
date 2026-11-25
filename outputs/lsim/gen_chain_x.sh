#!/bin/bash
# GPU 4: generative fine-tune from q64L step 1500, energy score, one-step + 300-frame (10 s) rollouts on c1_long (as q64X); then eval + renders.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
export CUDA_VISIBLE_DEVICES=4
R=cloth_g64X
$L -m lsim.train.cloth_train --data data/datasets/c1_long --steps 150 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 --kv_heads 4 --mlp 1536 \
  --lr 3e-5 --noise_lr 3e-4 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 --init outputs/lsim/cloth_q64L/step1500.pt \
  --roll 300 --roll_n 1 --gen energy --noise_dim 32 --val "data/datasets/c1_long_test/shard_*.npz" --log_every 10 --save_every 50 --keep \
  --out outputs/lsim/$R > outputs/lsim/$R.log 2>&1
for s in 149 100; do
  f=outputs/lsim/$R/step$s.pt; [ -f $f ] || continue
  $L scripts/l2_cloth_long.py --ckpt $f --data c1_long_test --frames 590 --samples 2 --tag long20_${R}_step$s > outputs/lsim/long20_${R}_step$s.log 2>&1
  $L scripts/l2_cloth_long.py --ckpt $f --data c1_test --samples 2 --tag long_${R}_step$s > outputs/lsim/long_${R}_step$s.log 2>&1
done
f=outputs/lsim/$R/step149.pt
for e in 0 5; do
  $L scripts/l2_render_cloth_samples.py export $f data/datasets/c1_long_test/shard_00000 $e 7 590 long20_${R}_samples > outputs/lsim/render_$R.log 2>&1
  d=outputs/lsim/render_cloth_model/shard_00000_e${e}_t7_long20_${R}_samples
  $B --background --python scripts/blender_render.py -- $d >> outputs/lsim/render_${R}_blender.log 2>&1
  $L scripts/l2_render_cloth_samples.py compose $d >> outputs/lsim/render_$R.log 2>&1
done
echo "[done] $(date -u)" >> outputs/lsim/$R.log
