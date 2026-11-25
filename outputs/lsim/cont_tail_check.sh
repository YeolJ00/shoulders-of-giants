#!/bin/bash
# Is the third segment's rollout tail a checkpoint swing? (2026-11-25) cloth_q64_ownup_cont2/step23999 blows up 21 of
# 336 held-out rollouts (area > 2x; mean strain 16.6%) where cont/step23999 blew up 6 (6.1%), with the fit better
# (0.486 vs 0.520) and the median rollout better. Rule: never judge on one checkpoint. Scored here, packed on one GPU:
# cont2/step20000 and cont/step20000 (the checkpoint before each segment's last).
# PREDICTION: the tail is real, not a swing -- cont2/step20000 also blows up 15 or more, cont/step20000 10 or fewer.
# READ: if cont2/step20000 has 10 or fewer, the 72k tail is one checkpoint's swing; if both 72k-segment checkpoints have
# 15+, longer training grows the tail (the fit and the long-horizon stability part ways), which the given arm
# (cloth_q64_ownup_cont2_given, same steps) may or may not share.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[tail] GPU $g $(date -u)"
for n in cloth_q64_ownup_cont2 cloth_q64_ownup_cont; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step20000.pt --tag cov_${n}_step20000 \
    > outputs/lsim/coverage_score_${n}_step20000.log 2>&1 &
done
wait
echo "[tail] scored $(date -u)"
