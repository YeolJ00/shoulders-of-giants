#!/bin/bash
# v6: v3 fine-tuned with a long-rollout loss on top of its own recipe (lsim.train.cloth_real.rollout_loss: two
# episodes per step rolled out alone $1 frames, every frame scored against the teacher, the graph cut every 10).
# Why: on PGND's scenes v3 drifts from its own teacher by 37 / 62 / 139 mm at 1 / 2 / 4 s (the teacher 16-17 mm from
# the cameras), and on c1's own episodes a 60-frame rollout ended 82 mm off.   train_cloth_v6.sh <frames> <tag> <gpu> [steps]
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=$3 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps ${4:-3000} --save_every 500 --keep --batch 4 --lr 3e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 \
  --init outputs/lsim/cloth_v3/last.pt --roll $1 --roll_n 2 --roll_every 10 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$2 > outputs/lsim/cloth_$2.log 2>&1
