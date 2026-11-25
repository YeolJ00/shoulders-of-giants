#!/bin/bash
# Drift diagnostic for each overfit variant as its checkpoint appears (A, B, E on GPU 2; D on GPU 1).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
D=/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/drift_diag.py
run() {   # variant gpu
    until [ -f outputs/lsim/of$1.pt ]; do sleep 20; done
    CUDA_VISIBLE_DEVICES=$2 $L -u $D outputs/lsim/of$1.pt 12 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_of$1.txt
    echo "[diag] $1 done"
}
run D 1 & run A 2 & run B 2 & run E 2 & wait
echo "[diag] all done"
