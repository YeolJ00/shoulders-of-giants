#!/bin/bash
# Held-out rollouts of the 2,000-episode pair at step 6,000 (2026-11-22): the one-frame lead of --own_up is -0.106
# [-0.132, -0.089] there (-0.214 at step 3,000). Its one-episode replays drift after ~2 s where the world-frame
# network's do not -- does that show on held-out episodes? 16 episodes per family, the same picks for both
# (l2_coverage_score's seeded choice), from frame 7 to the end. Both on GPU $1 (light jobs share a card).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[roll6000] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven/step6000.pt --n 20 --n_roll 16 \
  --tag cov_seven_step6000_roll > outputs/lsim/coverage_score_seven_6000_roll.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven_ownup/step6000.pt --n 20 --n_roll 16 \
  --tag cov_seven_ownup_step6000_roll > outputs/lsim/coverage_score_seven_ownup_6000_roll.log 2>&1 &
wait
echo "[roll6000] scored $(date -u)"
