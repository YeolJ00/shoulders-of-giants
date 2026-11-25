#!/bin/bash
# The rigid small-fit test's --plain arm on its own GPU (2026-11-22): sharing GPU 5 with the as-it-is arm both ran at
# 2.9 s a step (12,000 steps = 9.7 h; 22 GB each). Restarted here as rigid_fit_1_plainb; outputs/lsim/rigid_fit1.sh keeps
# the as-it-is arm (its own scoring of rigid_fit_1_plain.pt will find no file).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rigid1p] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1 --steps 12000 --pair_norm --coulomb --pf 6 \
  --noise 0.003,0.015 --mix --plain --out outputs/lsim/rigid_fit_1_plainb.pt > outputs/lsim/rigid_fit_1_plainb.log 2>&1
echo "[rigid1p] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/rigid_fit_1_plainb.pt --episodes 1 > outputs/lsim/small_fit_rigid_fit_1_plainb.log 2>&1
echo "[rigid1p] scored ($?) $(date -u)"
