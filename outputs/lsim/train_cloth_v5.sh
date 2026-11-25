#!/bin/bash
# The cloth model v5 on c1, from scratch: v3's recipe (layer 2, the old per-vertex chunk noise) with the mesh-edge
# length loss at weight 100 (v3: 1). Fine-tunes of v3 at step 2,000 (2026-10-22): one-frame / ten-frame median
# strain 0.89% / 3.69% at weight 1, 0.50% / 2.16% at 10, 0.29% / 0.93% at 100, the ten-frame position error
# unchanged; noise that moves each cloth as a body did nothing (0.77% / 3.71%). GPU from $2.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=${2:-4} /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${1:-24000} --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 100 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_v5 > outputs/lsim/cloth_v5.log 2>&1
