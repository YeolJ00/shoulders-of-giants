#!/bin/bash
# Soft-body dataset s1 (20,000) then s1_test (1,024) on GPU 2 (lsim.data.soft; ~2.2k episodes / GPU-hour, ~9.5 h).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_generate_soft.py --out data/datasets/s1 --episodes 20000 --worlds 64 > outputs/lsim/gen_s1.log 2>&1
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_generate_soft.py --out data/datasets/s1_test --episodes 1024 --split test --seed 1 > outputs/lsim/gen_s1_test.log 2>&1
echo "s1 done $(date -u)" >> outputs/lsim/gen_s1.log
