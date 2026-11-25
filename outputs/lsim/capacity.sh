#!/bin/bash
# Capacity arms beside v7 (v3's recipe 4x longer): v3's recipe for 24k steps from scratch, wider (v7w: d 256, GPU 2)
# and deeper (v7d: 12 layers, GPU 3); and v6k continued (v6k2: 3,000 steps at 3e-5, rollout weight 3, GPU 1).
# Each final checkpoint scored on the long horizon and PGND held-out on its own GPU.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
run(){ r=$1; g=$2; n=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g $L -u -m lsim.train.cloth_train --data data/datasets/c1 --steps $n --batch 4 --sig 1e-3 --pf 4 \
    --prob_slots --unc --accel --layer 2 --edge 1.0 "$@" --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$r \
    > outputs/lsim/cloth_$r.log 2>&1
  f=outputs/lsim/cloth_$r/last.pt; [ -f $f ] || return
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_cloth_long.py --ckpt $f --tag ${r}_last > outputs/lsim/long_${r}_last.log 2>&1
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag ${r}_last > outputs/lsim/pgnd_eval_${r}_last.log 2>&1
  echo "[capacity] $r evaluated $(date -u)"; }
run v7w 2 24000 --lr 1e-3 --d 256 --save_every 8000 --keep &
run v7d 3 24000 --lr 1e-3 --layers 12 --save_every 8000 --keep &
run v6k2 1 3000 --lr 3e-5 --init outputs/lsim/cloth_v6k/step1999.pt --roll 100 --roll_n 2 --roll_every 10 --roll_w 3 --save_every 1000 --keep &
wait
