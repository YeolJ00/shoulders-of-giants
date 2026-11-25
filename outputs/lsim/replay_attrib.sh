#!/bin/bash
# The small-fit test: why does the --own network replay worse than the world-frame one (2026-11-22)? One episode,
# chunk noise 1 mm + 4-frame pushforward, 4,000 steps:
#   cloth_fit_1_noaug_pf   world frame, --no_aug --all_past            frames 0.160   replay  8.7 /  4.8 / 17.2 mm
#   cloth_fit_1_own_pf     --own, random yaw + shift, random past      frames 0.103   replay 12.0 / 39.9 / 93.3 mm
# Three things differ: the frames, the augmentation, the random shorter past. One run per difference
# ($1 = name suffix, $2 = GPU, the rest = its flags), each scored by l2_small_fit.py:
#   cloth_fit_1_ownup_pf (outputs/lsim/ownup_pf1.sh)   --own_up, augmentation, random past   (the vertical)
#   cloth_fit_1_own_fix_pf    --own --no_aug --all_past      (own frames at one orientation, every past frame)
#   cloth_fit_1_ownup_all_pf  --own_up --all_past            (the candidate with every past frame, augmentation on)
# (the long-rollout fine-tune cloth_fit_1_own_roll was stopped at step ~30 for these: 42 s a step, 7 h for 600.)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
n=$1; g=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[attrib] $n on GPU $g: $* $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --save_every 1000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET "$@" --episodes 1 \
  --steps 4000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_$n \
  > outputs/lsim/cloth_fit_1_$n.log 2>&1
echo "[attrib] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_$n --episodes 1 > outputs/lsim/small_fit_1_$n.log 2>&1
echo "[attrib] $n scored ($?) $(date -u)"
