#!/bin/bash
# --own_up (2026-11-22): --own + the world-z part of each vector a cloth token reads and a force along world z.
# Why: --own fits what it trains on (one episode, the trainer as it is: 0.092 of constant velocity; the world-frame
# network 0.641) but on held-out cloth it does not cancel gravity (16 episodes at 2,000 steps, one frame on c1_test:
# vertical error 5.1 mm on held cloth, the world-frame network 1.5, the exact layer alone 5.7; the trainer's held-out
# val 4.6-5.4 mm against 1.5-1.6).
#   cloth_fit_16_ownup   cloth_fit_16's recipe (16 episodes, 8,000 steps, no noise, no pushforward) + --own_up
# on GPU $1 once it is empty twice a minute apart; scored by l2_small_fit.py. The read: the trainer's held-out val
# at steps 500 / 1,000 / 1,500 (cloth_fit_16: 2.46 / 1.58 / 1.63 mm; cloth_fit_16_own: 4.63 / 4.80 / 5.39), then its
# own 16 episodes (cloth_fit_16: 0.696).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[ownup] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 0 --pf 0 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --own_up --save_every 1000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 16 \
  --steps 8000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_16_ownup \
  > outputs/lsim/cloth_fit_16_ownup.log 2>&1
echo "[ownup] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_16_ownup --episodes 16 > outputs/lsim/small_fit_16_ownup.log 2>&1
echo "[ownup] scored ($?) $(date -u)"
