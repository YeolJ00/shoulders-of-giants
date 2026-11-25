#!/bin/bash
# Drift diagnostic on every new checkpoint of training v0 (GPU 1, ~10 min each).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
C=outputs/lsim/train_v0/last.pt
last=""
while ps aux | grep -q "[l]sim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 15000 --batch 16 --pair_norm"; do
    if [ -f $C ] && [ "$(stat -c %Y $C)" != "$last" ]; then
        sleep 30; last=$(stat -c %Y $C)
        step=$($L -c "import torch; print(torch.load('$C', map_location='cpu', weights_only=False)['step'])")
        echo "== step $step $(date -u +%H:%M)" >> outputs/lsim/diag_train_v0.txt
        CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_drift.py $C 12 free,tf 2>&1 | grep "^\[" >> outputs/lsim/diag_train_v0.txt
        echo "[ckpt_diag] step $step done"
    fi
    sleep 60
done
echo "[ckpt_diag] training ended"
