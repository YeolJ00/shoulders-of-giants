#!/bin/bash
# The cloth model v3 on c1: v2's recipe (layer 2) plus the mesh-edge length loss (--edge 1: each edge's length error
# against the teacher's next frame over its rest length; the teacher's towels strain 1e-4, v1's stretched 1% in one
# step and 5% in ten). The stretch is for the network to learn, not for the layer to project away. GPU from $2
# (default 3). Validation every 500 steps on held-out c1_test.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=${2:-3} /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${1:-24000} --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_v3 > outputs/lsim/cloth_v3.log 2>&1
