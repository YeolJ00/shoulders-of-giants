#!/bin/bash
# The small-fit test, the replay, world-frame control (2026-11-22): cloth_fit_1_noaug (one episode, no yaw/shift, every
# past frame, no noise, no pushforward: 0.10 of constant velocity on its own frames, its rollout explodes) with the
# full-data recipe's chunk noise and pushforward (--sig 1e-3 --pf 4). Does a network that has seen its own error replay
# the one episode it trained on? Independent of --own (outputs/lsim/own_pf_chain.sh runs that with the augmentation).
# GPU $1 once it is empty twice a minute apart; scored by l2_small_fit.py.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[noaug_pf] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --save_every 1000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 1 \
  --no_aug --all_past --steps 4000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_noaug_pf \
  > outputs/lsim/cloth_fit_1_noaug_pf.log 2>&1
echo "[noaug_pf] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_noaug_pf --episodes 1 > outputs/lsim/small_fit_1_noaug_pf.log 2>&1
echo "[noaug_pf] scored ($?) $(date -u)"
