#!/bin/bash
# After v0 and v0r are generated: dataset checks, then training of the model (GPU 1) and of the
# B2 baseline (GPU 2). Waits on the generator processes by name.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
while pgrep -f "scripts/l1_generate.py --out data/datasets/v0" > /dev/null || pgrep -f "kill -0" > /dev/null; do sleep 60; done
echo "[chain] data done $(date)"
ls data/datasets/v0 | grep -c '^shard_[0-9]*\.npz$'; ls data/datasets/v0r | grep -c '^shard_[0-9]*\.npz$'
CUDA_VISIBLE_DEVICES=1 setsid nohup $L -u -m lsim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test \
    --steps 30000 --batch 16 --out outputs/lsim/train_v0 > outputs/lsim/train_v0.log 2>&1 &
echo "[chain] model training started $(date)"
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_dataset_check.py --data data/datasets/v0 --sample 8 --repro 0 > outputs/lsim/dataset_check_v0.log 2>&1
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_gate.py --data data/datasets/v0 --shard 0 > outputs/lsim/gate_v0.log 2>&1
cp docs/results/l1_gate.json docs/results/l1_gate_v0.json
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_gate.py --data data/datasets/v0r --shard 0 > outputs/lsim/gate_v0r.log 2>&1
cp docs/results/l1_gate.json docs/results/l1_gate_v0r.json
echo "[chain] checks done $(date)"
CUDA_VISIBLE_DEVICES=2 setsid nohup $L -u -m lsim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test \
    --steps 30000 --batch 16 --direct --out outputs/lsim/train_b2 > outputs/lsim/train_b2.log 2>&1 &
echo "[chain] B2 training started $(date)"
