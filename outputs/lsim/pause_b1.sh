#!/bin/bash
# Pause baseline B1 once its step-2000 checkpoint is on disk (resume later with --resume), freeing
# GPU 2 for the v1 (Coulomb default) tests.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
C=outputs/lsim/train_b1/last.pt
t0=$(stat -c %Y $C)
until grep -q "^step   2000 " outputs/lsim/train_b1.log && [ "$(stat -c %Y $C)" != "$t0" ]; do sleep 20; done
sleep 30                                                       # let the write finish
step=$(/home/jooyeolyun/anaconda3/envs/lsim/bin/python -c "import torch; print(torch.load('$C', map_location='cpu', weights_only=False)['step'])")
P=$(ps aux | grep "[l]sim.train.train.*--arch gns --out outputs/lsim/train_b1" | awk '{print $2}')
kill $P; echo "[pause_b1] B1 stopped; checkpoint step $step $(date -u)"
