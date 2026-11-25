#!/bin/bash
# c1 regeneration, both halves on GPU 3 (GPU 4 is for the cloth model): the even half (pid in $1) is already running
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
P0=$1
echo "[c1] odd half start $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 20000 --worlds 64 --seed 0 --part 1/2 > outputs/lsim/gen_c1v2_p1.log 2>&1
while kill -0 $P0 2>/dev/null; do sleep 30; done
echo "[c1] train done $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1_test --episodes 1024 --worlds 64 --seed 1 --split test --part 0/1 > outputs/lsim/gen_c1v2_test.log 2>&1
echo "[c1] test done $(date -u)"
