#!/bin/bash
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
run() { until [ -f outputs/lsim/of$1.pt ]; do sleep 20; done
        CUDA_VISIBLE_DEVICES=$2 $L -u scripts/l2_drift.py outputs/lsim/of$1.pt 12 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_of$1.txt; echo "[diag] $1 done"; }
run H 2 & run I 1 & wait
echo "[diag] HI done"
