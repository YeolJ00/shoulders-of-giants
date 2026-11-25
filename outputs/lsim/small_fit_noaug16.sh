#!/bin/bash
# The small-fit test, 16 episodes without the augmentation (2026-11-22): cloth_fit_16's recipe (0.70 of constant
# velocity on its own 16 episodes after 8,000 steps with the trainer's random yaw + shift and random shorter past) with
# --no_aug --all_past. GPU 7, empty twice a minute apart. Scored by l2_small_fit.py.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=7
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 120; done
echo "[noaug16] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 0 --pf 0 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --save_every 1000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 16 \
  --no_aug --all_past --steps 8000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_16_noaug \
  > outputs/lsim/cloth_fit_16_noaug.log 2>&1
echo "[noaug16] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_16_noaug --episodes 16 > outputs/lsim/small_fit_16_noaug.log 2>&1
echo "[noaug16] scored ($?) $(date -u)"
