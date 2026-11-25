#!/bin/bash
# v7 = v3's recipe 4x longer (96k steps, ~32 h on one GPU), from scratch: the capacity / precision test. v3's clean
# step was still falling as OneCycle annealed (0.65 mm at 17.5k, 0.59 at 23.5k), and every long-rollout fine-tune of
# it (v6e: c1_test 30 / 85 / 107 mm at 1 / 2 / 3.8 s; still 57 / 144 / 168) plateaus; the teacher's material spread
# explains a third of that at most. Does a more precise step lower the long-horizon floor? Starts on GPU 0 when r12
# ends; each 8k checkpoint scored with scripts/l2_cloth_long.py on the same GPU.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until [ -f docs/results/l2_real_cloth_pgnd_r12_step1000.json ]; do sleep 120; done
CUDA_VISIBLE_DEVICES=0 $L -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --steps 96000 --save_every 8000 --keep --batch 4 --lr 1e-3 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_v7 > outputs/lsim/cloth_v7.log 2>&1 &
for s in $(seq 8000 8000 88000); do
  until [ -f outputs/lsim/cloth_v7/step$s.pt ] || grep -q "^Traceback" outputs/lsim/cloth_v7.log; do sleep 300; done
  [ -f outputs/lsim/cloth_v7/step$s.pt ] || break; sleep 30
  CUDA_VISIBLE_DEVICES=0 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_v7/step$s.pt --tag v7_step$s > outputs/lsim/long_v7_step$s.log 2>&1
  echo "[v7] step $s evaluated $(date -u)"
done
wait
