#!/bin/bash
# The small-fit test, the replay, trained on its own replays (2026-11-22). cloth_fit_1_own_pf (one episode, --own,
# chunk noise + pushforward) fits the frames to 0.10 of constant velocity and replays the episode to 12 / 40 / 93 mm at
# 1 s / 2 s / 3.8 s; rendered, it follows the teacher for about a second, then the cloth stretches (area 1.3x at
# 1.2 s) and crumples (3.1x at the end): one episode teaches nothing about the states a frame-by-frame drift reaches.
# The long-rollout loss (v6e's recipe: lr 3e-5, 100-frame rollouts of the training episode from frames 7-19, the
# graph cut every 10 frames, weight 3, the material given) trains on exactly those states:
#   cloth_fit_1_own_roll   from cloth_fit_1_own_pf/step3999.pt, 600 steps
# GPU $1 once it is empty twice a minute apart; scored by l2_small_fit.py (the replay with a window per frame, and
# the one-window rollout the loss itself uses beside it).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[roll] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --own --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 1 \
  --init outputs/lsim/cloth_fit_1_own_pf/step3999.pt --lr 3e-5 --steps 600 --save_every 200 --log_every 10 \
  --roll 100 --roll_n 1 --roll_every 10 --roll_w 3 --roll_given 1.0 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_own_roll > outputs/lsim/cloth_fit_1_own_roll.log 2>&1
echo "[roll] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_own_roll --episodes 1 > outputs/lsim/small_fit_1_own_roll.log 2>&1
echo "[roll] scored ($?) $(date -u)"
