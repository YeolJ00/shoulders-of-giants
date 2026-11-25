#!/bin/bash
# Resume a fine-tune of train_cloth_ft.sh from its last.pt (saved every 2,000 steps) on another GPU, appending to its
# log (2026-10-22: v4c moved off the GPU it shared with v4r).   outputs/lsim/train_cloth_ft_resume.sh <name> <steps> <gpu> [args...]
cd /home/nas5/jooyeolyun/repos/simulation-assestization
NAME=$1; STEPS=$2; GPU=$3; shift 3
echo "[resume] $(date -u) on GPU $GPU" >> outputs/lsim/cloth_$NAME.log
CUDA_VISIBLE_DEVICES=$GPU /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps $STEPS --batch 4 --lr 5e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --init outputs/lsim/cloth_v3/last.pt --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$NAME --resume "$@" \
  >> outputs/lsim/cloth_$NAME.log 2>&1
