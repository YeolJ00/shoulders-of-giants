#!/bin/bash
# Baseline B2 for cloth on GPU 3 once c1_test is generated: the main run's recipe with direct accelerations
# (no gravity, drag, zero-sum or contact default in an exact layer), so the two isolate the layer.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
until grep -q "\[c1\] test done" outputs/lsim/gen_c1v3_driver.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=3 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${1:-24000} --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel --direct \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_b2 > outputs/lsim/cloth_b2.log 2>&1
