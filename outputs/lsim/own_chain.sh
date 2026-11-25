#!/bin/bash
# --own (cloth tokens in their own frames, cloth_qwen._forward_own; 2026-11-22) against the world-frame network, the
# trainer as it is (random yaw + shift, random past). In l2_overfit_batch, 8 windows of one episode re-yawed every step:
# world frame 0.71-0.72 of constant velocity at step 400 (stuck to 1,200 with the shift), --own 0.15 (the un-yawed
# curve). Three runs, one GPU each (three found empty three times over 2.5 minutes):
#   cloth_fit_1_own     cloth_fit_1 + --own   (1 episode, 4,000 steps; cloth_fit_1: 0.64 on its own episode)
#   cloth_fit_16_own    cloth_fit_16 + --own  (16 episodes, 8,000 steps; cloth_fit_16: 0.70)
#   cloth_cov_seven_own the coverage study's "seven" arm + --own (2,000 episodes of all seven families, 12,000 steps,
#                       the full recipe), scored like every arm by l2_coverage_score.py: the held-out comparison
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' '); sleep 75; b=$(empty_gpus | tr '\n' ' '); sleep 75; c=$(empty_gpus | tr '\n' ' ')
  ab=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort))
  G=$(comm -12 <(echo "$ab") <(echo $c | tr ' ' '\n' | sort) | sort -n | head -3 | tr '\n' ' ')
  [ $(echo $G | wc -w) -ge 3 ] || { G=""; sleep 90; }
done
GPUS=($G)
echo "[own] GPUs $G $(date -u)"
BASE="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel
  --layer 2 --edge 1.0 --side_apart --own --keep --data data/datasets/c1"
small() {   # $1 episodes, $2 steps, $3 gpu
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$3 $L -m lsim.train.cloth_train $BASE --sig 0 --pf 0 \
    --shards 1 --episodes $1 --steps $2 --save_every 1000 --val "data/datasets/c1_test/shard_*.npz" \
    --out outputs/lsim/cloth_fit_$1_own > outputs/lsim/cloth_fit_$1_own.log 2>&1
  echo "[own] $1 episodes trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$3 $L scripts/l2_small_fit.py --run cloth_fit_$1_own --episodes $1 > outputs/lsim/small_fit_$1_own.log 2>&1
  echo "[own] $1 episodes scored ($?) $(date -u)"
}
small 1 4000 ${GPUS[0]} &
small 16 8000 ${GPUS[1]} &
( g=${GPUS[2]}
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $BASE --sig 1e-3 --pf 4 \
    --steps 12000 --save_every 3000 --subset outputs/lsim/coverage_subsets/seven.json \
    --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_cov_seven_own > outputs/lsim/cloth_cov_seven_own.log 2>&1
  echo "[own] seven trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven_own/step11999.pt \
    --tag cov_seven_own_step11999 > outputs/lsim/coverage_score_seven_own.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven_own/step9000.pt --n_roll 0 \
    --tag cov_seven_own_step9000 > outputs/lsim/coverage_score_seven_own_9000.log 2>&1 &
  wait
  echo "[own] seven scored $(date -u)" ) &
wait
echo "[own] all done $(date -u)"
