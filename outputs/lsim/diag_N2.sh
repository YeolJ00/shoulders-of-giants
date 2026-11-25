#!/bin/bash
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
S=/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad
until [ -f outputs/lsim/ofN2.pt ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_drift.py outputs/lsim/ofN2.pt 12 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_ofN2.txt
CUDA_VISIBLE_DEVICES=1 $L -u $S/rest_breakdown.py outputs/lsim/ofN2.pt 2>&1 | grep -E "^  " >> outputs/lsim/diag_ofN2.txt
echo "[diag] N2 done"
