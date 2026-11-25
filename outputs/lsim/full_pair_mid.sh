#!/bin/bash
# The full-data pair mid-run, on more than one checkpoint (2026-11-23). At step 8,000 cloth_q64_ownup is level with
# cloth_q64 on one frame (0.816 vs 0.809, +0.007 [-0.012, +0.027]) after leading by 0.075 at 4,000 -- but mid-schedule
# checkpoints swing: the trainer's 64-window val read 0.675 / 0.924 mm at 7,500 / 8,000 for --own_up and 1.030 / 0.728
# for q64, so 8,000 pairs a poor checkpoint with a good one. This scores one network ($1 = cloth_q64 or
# cloth_q64_ownup; tag prefix $2 = cov_full_q64 or cov_full_q64_ownup) on GPU $3 (empty twice a minute apart):
# one frame at step 6,000, then one frame + 48 held-out rollouts a family at step 8,000 (tag ..._step8000_roll).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
n=$1; t=$2; g=$3
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[mid] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step6000.pt --n_roll 0 \
  --tag ${t}_step6000 > outputs/lsim/coverage_score_${t}_6000.log 2>&1
echo "[mid] $n 6000 ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step8000.pt \
  --tag ${t}_step8000_roll > outputs/lsim/coverage_score_${t}_8000_roll.log 2>&1
echo "[mid] $n 8000 rollouts ($?) $(date -u)"
