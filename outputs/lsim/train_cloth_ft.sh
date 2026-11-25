#!/bin/bash
# Fine-tune the cloth model v3 for a few thousand steps with one change, against a control with none:
#   outputs/lsim/train_cloth_ft.sh <out name> <steps> <gpu> [extra trainer args...]
# v4r: --noise rigid (each cloth moved as a body, 1 mm + 3e-3 rad per frame, plus 2e-5 m white): the per-vertex 1 mm
# noise strained the inputs' 1 cm edges by 7% median (24% p90), 10,000x the teacher's training frames.
# v4c: the control, v3's own noise. Both from v3's last.pt, lr 5e-4, validation every 500 steps on held-out c1_test.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
NAME=$1; STEPS=$2; GPU=$3; shift 3
CUDA_VISIBLE_DEVICES=$GPU /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps $STEPS --batch 4 --lr 5e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --init outputs/lsim/cloth_v3/last.pt --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$NAME "$@" \
  > outputs/lsim/cloth_$NAME.log 2>&1
