#!/bin/bash
cd /home/nas5/jooyeolyun/repos/simulation-assestization
until [ -f outputs/lsim/ofM.pt ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u scripts/l2_drift.py outputs/lsim/ofM.pt 12 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_ofM.txt
echo "[diag] M done"
