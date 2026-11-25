#!/bin/bash
# The two contact defaults alone (untrained network: force heads start at zero), on the real
# Physics 101 slides and on v0r's ramps -- what each prior does before any learning.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for n in of0C of0S; do
    CUDA_VISIBLE_DEVICES=4 $L -u scripts/l2_phys101.py --ckpt outputs/lsim/$n.pt --out outputs/lsim/phys101_$n.json > outputs/lsim/phys101_$n.log 2>&1
    CUDA_VISIBLE_DEVICES=4 $L -u scripts/l2_drift.py outputs/lsim/$n.pt 12 free,tf data/datasets/v0r 2>&1 | grep "^\[" > outputs/lsim/drift_$n.txt
    echo "[priors] $n scored $(date -u)"
done
