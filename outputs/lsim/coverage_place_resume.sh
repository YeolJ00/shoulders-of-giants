#!/bin/bash
# The coverage study's lift_place arm, resumed (2026-11-22): stopped by mistake at step 6,750 to free a GPU the small
# fit test turned out not to need; resumed from its step-6,000 save (model, optimiser, schedule) with the chain's own
# command + --resume, then scored as the chain scores every arm. Waits for GPU 1 empty twice a minute apart.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 120; done
echo "[resume] place on GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 12000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4
  --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --save_every 3000 --keep --data data/datasets/c1"
arm=place
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --resume \
  --subset outputs/lsim/coverage_subsets/$arm.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/cloth_cov_$arm >> outputs/lsim/cloth_cov_$arm.log 2>&1
echo "[resume] $arm trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_$arm/step11999.pt \
  --tag cov_${arm}_step11999 > outputs/lsim/coverage_score_$arm.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_$arm/step9000.pt --n_roll 0 \
  --tag cov_${arm}_step9000 > outputs/lsim/coverage_score_${arm}_8999.log 2>&1 &
wait
echo "[resume] $arm scored $(date -u)"
