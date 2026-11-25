#!/bin/bash
# 2026-10-22: after the fine-tune chain (ft_chain.sh, GPU 3) ends, the cloth models on real cloth (PhysTwin, converted
# by scripts/l1_real_cloth.py): v3 and the two fine-tunes, on GPU 3.   nohup outputs/lsim/real_eval_chain.sh <ft_chain pid> &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
while kill -0 $1 2>/dev/null; do sleep 60; done
for t in v3 v4r v4c; do
  CUDA_VISIBLE_DEVICES=3 /home/jooyeolyun/anaconda3/envs/lsim/bin/python scripts/l2_real_cloth_eval.py \
    --ckpt outputs/lsim/cloth_$t/last.pt --tag $t > outputs/lsim/real_eval_$t.log 2>&1
  echo "[real eval] $t done $(date -u)"
done
