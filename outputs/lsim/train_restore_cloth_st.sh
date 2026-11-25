#!/bin/bash
# The exact layer's edge / volume restoration (cloth_net.restore_shape, --restore 1; 2026-11-03), fine-tuned:
#   GPU 2: cloth, as GPU 3 but the restoration straight-through in the long-rollout loss (st_roll: its gradient through 300-frame rollouts reached 1e6-3e7)
#   GPU 4: mixed -- the 64M mixed model (m64 step 23,999) on c1 + r1 + s1 with 30-frame rollouts, as cloth_m64R was run
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=2 nohup $L -m lsim.train.cloth_train --data data/datasets/c1_long --steps 200 --batch 4 --d 512 --layers 16 \
  --arch qwen --heads 8 --kv_heads 4 --mlp 1536 --lr 3e-5 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 \
  --val 'data/datasets/c1_long_test/shard_*.npz' --init outputs/lsim/cloth_q64L/step1500.pt --roll 300 --roll_w 1 --roll_n 2 \
  --roll_every 10 --save_every 50 --log_every 10 --keep --restore st_roll=1 --out outputs/lsim/cloth_q64XS > outputs/lsim/cloth_q64XS.log 2>&1 &
