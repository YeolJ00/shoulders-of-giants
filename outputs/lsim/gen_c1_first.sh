#!/bin/bash
# The first group of four 64-episode cloth shards (sorted by stiffness), GPUs 3 and 4, then the gate on it.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 256 --worlds 64 --seed 0 --part 0/2 > outputs/lsim/gen_c1_first_p0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 256 --worlds 64 --seed 0 --part 1/2 > outputs/lsim/gen_c1_first_p1.log 2>&1 &
wait
echo "[c1 first] generated $(date -u)"
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l1_cloth_gate.py --data data/datasets/c1 --shards 0,1,2,3 --per_shard 16 > outputs/lsim/gate_c1.log 2>&1
echo "[c1 first] gated $(date -u)"
