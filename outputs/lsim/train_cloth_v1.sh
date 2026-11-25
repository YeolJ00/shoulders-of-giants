#!/bin/bash
# The cloth model v1 on c1 (picks up shards as generation finishes), GPU 4: lr 1e-3, chunk noise 1 mm per frame,
# pushforward up to 4 own frames, acceleration inputs, the slot posterior and the one-step uncertainty head;
# validation every 500 steps on held-out c1_test once it exists.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=4 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${1:-24000} --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_v1 > outputs/lsim/cloth_v1.log 2>&1
