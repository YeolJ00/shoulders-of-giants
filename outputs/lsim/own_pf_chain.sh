#!/bin/bash
# The small-fit test, the replay (2026-11-22). cloth_fit_1_noaug (the trainer on ONE episode, no yaw/shift, no noise, no
# pushforward) fits that episode's frames to 0.10 of constant velocity (0.35 / 0.20 / 0.13 / 0.10 at 1k / 2k / 3k / 4k
# steps; cloth_fit_1 with the augmentation: 0.64) and its rollout of it still explodes (216 / 334 / 691 mm at 1 s / 2 s
# / end, a still cloth 283 / 311 / 272; strain 4,855%): trained on clean states only, it has never seen its own error.
# Here the small fit with the recipe that keeps the full-data network's rollouts together (chunk noise 1 mm, 4-frame
# pushforward) and --own, the trainer as it is (random yaw + shift, random past):
#   cloth_fit_1_own_pf    1 episode, 4,000 steps     GPU $1
#   cloth_fit_16_own_pf   16 episodes, 8,000 steps   GPU $2
# each on its GPU once that is empty twice a minute apart; scored by l2_small_fit.py (its own episodes one frame ahead,
# and its rollout of them beside a still cloth).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --own --save_every 1000 --keep --data data/datasets/c1 --shards 1"
small() {   # $1 episodes, $2 steps, $3 gpu
  u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$3 '$1==g{print $2}')
  busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
  while busy || { sleep 60; busy; }; do sleep 60; done
  echo "[own_pf] $1 episodes on GPU $3 $(date -u)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$3 $L -m lsim.train.cloth_train $NET --episodes $1 \
    --steps $2 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_$1_own_pf \
    > outputs/lsim/cloth_fit_$1_own_pf.log 2>&1
  echo "[own_pf] $1 episodes trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$3 $L scripts/l2_small_fit.py --run cloth_fit_$1_own_pf --episodes $1 > outputs/lsim/small_fit_$1_own_pf.log 2>&1
  echo "[own_pf] $1 episodes scored ($?) $(date -u)"
}
small 1 4000 $1 &
small 16 8000 $2 &
wait
echo "[own_pf] all done $(date -u)"
