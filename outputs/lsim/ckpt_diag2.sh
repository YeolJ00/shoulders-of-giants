#!/bin/bash
# Drift diagnostic (vertex error) on every new checkpoint of training v0, on GPU 2 (rank 1 has the
# headroom; rank 0 also runs validation).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
C=outputs/lsim/train_v0/last.pt
until ps aux | grep -q "[l]sim.train.train --data data/datasets/v0,data/datasets/v0r.*--out outputs/lsim/train_v0"; do sleep 30; done
last=$( [ -f $C ] && stat -c %Y $C )
while ps aux | grep -q "[l]sim.train.train --data data/datasets/v0,data/datasets/v0r.*--out outputs/lsim/train_v0"; do
    if [ -f $C ] && [ "$(stat -c %Y $C)" != "$last" ]; then
        sleep 30; last=$(stat -c %Y $C)
        step=$($L -c "import torch; print(torch.load('$C', map_location='cpu', weights_only=False)['step'])")
        echo "== step $step $(date -u +%H:%M)" >> outputs/lsim/diag_train_v0.txt
        CUDA_VISIBLE_DEVICES=2 $L -u scripts/l2_drift.py $C 12 free,tf 2>&1 | grep "^\[" >> outputs/lsim/diag_train_v0.txt
        echo "[ckpt_diag] step $step done"
    fi
    sleep 60
done
echo "[ckpt_diag] training ended"
