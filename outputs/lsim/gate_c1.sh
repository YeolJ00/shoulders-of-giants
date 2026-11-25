#!/bin/bash
# The cloth identifiability gate: 16 episodes from each shard of the first stiffness group (packed on GPU 4).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=4 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u scripts/l1_cloth_gate.py --data data/datasets/c1 --shards 0,1,2,3 --per_shard 16 > outputs/lsim/gate_c1.log 2>&1
echo "[gate c1] done $(date -u)" >> outputs/lsim/gate_c1.log
