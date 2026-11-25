#!/bin/bash
# The small-fit test, the replay, three times the training (2026-11-22). cloth_fit_1_own_pf (one episode, --own, chunk
# noise 1 mm + 4-frame pushforward, the trainer as it is, 4,000 steps) fits its episode's frames to 0.40 / 0.21 / 0.13 /
# 0.10 of constant velocity at 1k / 2k / 3k / 4k steps -- still falling when the schedule ended -- and replays it
# (a window per frame) to 12 / 40 / 93 mm at 1 s / 2 s / 3.8 s (a still cloth 283 / 311 / 272; strain 140% at the
# end): about the sum of its one-frame errors. Here the same with --own_up (the candidate) and 12,000 steps:
#   cloth_fit_1_ownup_pf12k   does the frame fit keep falling, and the replay with it?
# GPU $1 once it is empty twice a minute apart; scored by l2_small_fit.py (every 2,000 steps one frame; the last
# checkpoint replayed).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[long] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --own_up --save_every 2000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 1 \
  --steps 12000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_ownup_pf12k \
  > outputs/lsim/cloth_fit_1_ownup_pf12k.log 2>&1
echo "[long] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_ownup_pf12k --episodes 1 > outputs/lsim/small_fit_1_ownup_pf12k.log 2>&1
echo "[long] scored ($?) $(date -u)"
