#!/bin/bash
# c1 with the final cloth teacher (per-mass contacts, shared implicit damping, edge-edge self-contact, clearance,
# guarded placing; fp32 positions): GPU 3, the train split then the held-out split
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
echo "[c1] start $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 20000 --worlds 64 --seed 0 --part 0/1 > outputs/lsim/gen_c1v3.log 2>&1
echo "[c1] train done $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1_test --episodes 1024 --worlds 64 --seed 1 --split test --part 0/1 > outputs/lsim/gen_c1v3_test.log 2>&1
echo "[c1] test done $(date -u)"
