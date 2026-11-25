#!/bin/bash
# The exact layer's edge / volume restoration (cloth_net.restore_shape, --restore 1; 2026-11-03), fine-tuned:
#   GPU 3: cloth -- the long-rollout 64M (q64L step 1,500) on c1_long with 300-frame rollouts, as cloth_q64X was run
#   GPU 4: mixed -- the 64M mixed model (m64 step 23,999) on c1 + r1 + s1 with 30-frame rollouts, as cloth_m64R was run
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=4 nohup $L -m lsim.train.cloth_train --data data/datasets/c1,data/datasets/r1,data/datasets/s1 --data_w 0.5,0.25,0.25 \
  --steps 2000 --batch 4 --d 512 --layers 16 --arch qwen3d --heads 8 --kv_heads 4 --mlp 1536 --lr 3e-5 --sig 1e-3 --pf 4 \
  --prob_slots --unc --accel --layer 2 --edge 10 --noise_rel \
  --val 'data/datasets/c1_test/shard_*.npz,data/datasets/r1_test/shard_*.npz,data/datasets/s1_test/shard_*.npz' \
  --init outputs/lsim/cloth_m64/step23999.pt --roll 30 --roll_w 1 --roll_n 2 --roll_every 10 --save_every 500 --log_every 50 \
  --keep --restore 1 --out outputs/lsim/cloth_m64P > outputs/lsim/cloth_m64P.log 2>&1 &
