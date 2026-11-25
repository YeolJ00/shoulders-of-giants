#!/bin/bash
# Claim 3's curve, parallel (the sequential chain was stopped after n=10): n=30 keeps training on GPU 7 and is scored
# when done; n=100 on GPU 3. Plus r7: r5's recipe from v6e step 500 (the horizon fix) on all 285 windows, GPU 1.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
ev(){ until [ -f outputs/lsim/cloth_$1/step1000.pt ] || grep -q Traceback outputs/lsim/cloth_$1.log; do sleep 60; done
      CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$1/step1000.pt --tag $1 > outputs/lsim/pgnd_eval_$1.log 2>&1
      echo "[claim3] $1 evaluated $(date -u)"; }
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_v6a/step500.pt --out outputs/lsim/cloth_r5n100 \
  --back_w 1 --strain_w 1 --n_real 100 > outputs/lsim/cloth_r5n100.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_v6e/step500.pt --out outputs/lsim/cloth_r7 \
  --back_w 1 --strain_w 1 > outputs/lsim/cloth_r7.log 2>&1 &
ev r5n30 7 & ev r5n100 3 & ev r7 1 & wait
