#!/bin/bash
# 2026-10-22 (replaces ft_chain.sh and real_eval_chain.sh after v4c moved to GPU 4): when each fine-tune ends, its
# evaluation (eval_cloth_ft.sh: rollouts, stretch, renders) and its real-cloth evaluation, on its own GPU.
#   nohup outputs/lsim/ft_chain2.sh <v4r pid> <v4c pid> &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
run() {   # name pid gpu
  while kill -0 $2 2>/dev/null; do sleep 60; done
  echo "[ft chain] $1 ended $(date -u)"
  outputs/lsim/eval_cloth_ft.sh outputs/lsim/cloth_$1/last.pt $1 $3
  CUDA_VISIBLE_DEVICES=$3 $L scripts/l2_real_cloth_eval.py --ckpt outputs/lsim/cloth_$1/last.pt --tag $1 > outputs/lsim/real_eval_$1.log 2>&1
  echo "[ft chain] $1 evaluated $(date -u)"
}
run v4r $1 3 &
run v4c $2 4 &
wait
echo "[ft chain] all done $(date -u)"
