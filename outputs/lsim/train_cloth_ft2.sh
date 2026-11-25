#!/bin/bash
# Fine-tunes of v3 with one change each (2026-10-22, on otherwise idle GPUs), as train_cloth_ft.sh but with the edge
# loss's weight as an argument:   outputs/lsim/train_cloth_ft2.sh <name> <steps> <gpu> <edge weight> [args...]
cd /home/nas5/jooyeolyun/repos/simulation-assestization
NAME=$1; STEPS=$2; GPU=$3; EDGE=$4; shift 4
CUDA_VISIBLE_DEVICES=$GPU /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps $STEPS --batch 4 --lr 5e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge $EDGE \
  --init outputs/lsim/cloth_v3/last.pt --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$NAME "$@" \
  > outputs/lsim/cloth_$NAME.log 2>&1
