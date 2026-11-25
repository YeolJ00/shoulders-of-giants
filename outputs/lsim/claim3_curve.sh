#!/bin/bash
# Claim 3's curve: r5's recipe (v6a step 500, back unclipped, strain 1) on 10 / 30 / 100 of the 285 real training
# windows, each scored on PGND's held-out recordings; 285 = cloth_r5. One after another on GPU 7.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for n in 10 30 100; do
  CUDA_VISIBLE_DEVICES=7 $L -u scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_v6a/step500.pt --out outputs/lsim/cloth_r5n$n \
    --back_w 1 --strain_w 1 --n_real $n > outputs/lsim/cloth_r5n$n.log 2>&1
  CUDA_VISIBLE_DEVICES=7 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_r5n$n/last.pt --tag r5n$n > outputs/lsim/pgnd_eval_r5n$n.log 2>&1
  echo "[claim3] n=$n evaluated $(date -u)"
done
