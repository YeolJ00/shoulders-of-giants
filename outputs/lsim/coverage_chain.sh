#!/bin/bash
# The coverage study (claim 1, 2026-11-22; scripts/l1_coverage_subsets.py): six 64M networks from scratch, the same
# recipe and compute (cloth_q64's recipe + --side_apart, 12,000 steps x 1 GPU x batch 4 = 48k windows), trained on
# 2,000 episodes of one gripper family (drag; lift_place), two, five, 8,000 of the same five, and 2,000 of all seven.
# Fold and shake are in no arm but the last. Each scored when it finishes, on its own GPU, by
# scripts/l2_coverage_score.py (c1_test's single-family episodes, family by family): step 11999 in full, step 9000 one
# frame only (never judge on one checkpoint). The full-data network (2x q64's training) scored once as the reference.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
ARMS="drag place drag_place five five_8k seven"
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' ')
  sleep 60
  b=$(empty_gpus | tr '\n' ' ')
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -6 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 6 ] && G="$both" || sleep 120
done
GPUS=($G)
echo "[coverage] arms $ARMS on GPUs $G $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 12000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4
  --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --save_every 3000 --keep --data data/datasets/c1"
i=0
for arm in $ARMS; do
  g=${GPUS[$i]}
  ( PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET \
      --subset outputs/lsim/coverage_subsets/$arm.json --val "data/datasets/c1_test/shard_*.npz" \
      --out outputs/lsim/cloth_cov_$arm > outputs/lsim/cloth_cov_$arm.log 2>&1
    echo "[coverage] $arm trained ($?) $(date -u)"
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_$arm/step11999.pt \
      --tag cov_${arm}_step11999 > outputs/lsim/coverage_score_$arm.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_$arm/step9000.pt --n_roll 0 \
      --tag cov_${arm}_step9000 > outputs/lsim/coverage_score_${arm}_8999.log 2>&1 &
    if [ $arm = drag ]; then
      CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt \
        --tag cov_full_q64_cont > outputs/lsim/coverage_score_full.log 2>&1 &
    fi
    wait
    echo "[coverage] $arm scored $(date -u)" ) &
  i=$((i + 1))
done
wait
echo "[coverage] all done $(date -u)"
