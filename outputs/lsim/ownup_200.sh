#!/bin/bash
# Between memorising and generalising (2026-11-22): --own_up on 200 episodes (outputs/lsim/coverage_subsets/
# seven_200.json: 200 of the seven arm's 2,000, all seven gripper families), the coverage arms' recipe and compute
# (12,000 steps x 1 GPU x batch 4 = 48k windows over ~21k: each window seen ~2.3 times; the 2,000-episode arms see each
# 0.23 times, one episode took ~150 views a window to 0.10). The read: its own 200 episodes against held-out ones
# (l2_coverage_score --data c1 --subset ... against c1_test), beside the 2,000-episode run's -- does the candidate
# network open a train / held-out gap where the world-frame one never did (train = test at every size)? That gap, or
# its absence, is what "scale beyond 20k?" turns on.
#   cloth_cov200_ownup    GPU $1 once it is empty twice a minute apart
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[own200] GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --subset outputs/lsim/coverage_subsets/seven_200.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/cloth_cov200_ownup > outputs/lsim/cloth_cov200_ownup.log 2>&1
echo "[own200] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov200_ownup/step11999.pt --n_roll 0 \
  --tag cov200_ownup_step11999 > outputs/lsim/coverage_score_200_ownup.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov200_ownup/step11999.pt --n_roll 0 \
  --data c1 --subset outputs/lsim/coverage_subsets/seven_200.json --tag cov200_ownup_step11999_train \
  > outputs/lsim/coverage_score_200_ownup_train.log 2>&1 &
wait
echo "[own200] scored $(date -u)"
