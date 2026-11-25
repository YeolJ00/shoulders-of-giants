#!/bin/bash
# The cloth model v2 on c1, GPU 4: v1's recipe (lr 1e-3, chunk noise 1 mm per frame, pushforward up to 4 own frames,
# acceleration inputs, the slot posterior and the one-step uncertainty head) under exact layer 2: friction capped by
# the surface's whole normal impulse (the network's contact push included) and the exact planes pushing sunk
# vertices out (lsim.model.cloth_net, 2026-10-20). Validation every 500 steps on held-out c1_test.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=4 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${1:-24000} --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_v2 > outputs/lsim/cloth_v2.log 2>&1
