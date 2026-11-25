#!/bin/bash
# After v0 and v0r are generated: dataset checks and the identifiability gate on the real
# datasets (GPU 2). Training is launched separately, once the step-loss settings are chosen.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
while pgrep -f "scripts/l1_generate.py --out data/datasets/v0r" > /dev/null; do sleep 60; done
echo "[chain] data done $(date)"
ls data/datasets/v0 | grep -c '^shard_[0-9]*\.npz$'; ls data/datasets/v0r | grep -c '^shard_[0-9]*\.npz$'
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_dataset_check.py --data data/datasets/v0 --sample 8 --repro 0 > outputs/lsim/dataset_check_v0.log 2>&1
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_gate.py --data data/datasets/v0 --shard 0 > outputs/lsim/gate_v0.log 2>&1
cp docs/results/l1_gate.json docs/results/l1_gate_v0.json
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_gate.py --data data/datasets/v0r --shard 0 > outputs/lsim/gate_v0r.log 2>&1
cp docs/results/l1_gate.json docs/results/l1_gate_v0r.json
cp docs/results/l1_gate_smoke2.json docs/results/l1_gate.json
echo "[chain] checks done $(date)"
