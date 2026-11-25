#!/bin/bash
# 2026-10-22: the edge-weight fine-tunes' final evaluations (both on GPU 5: evaluations are light), then v4e1000's
# when it ends (GPU 4).   nohup outputs/lsim/eval_e_chain.sh <v4e1000 pid> &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
ev() {   # name gpu
  outputs/lsim/eval_cloth_ft.sh outputs/lsim/cloth_$1/last.pt $1 $2
  CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_real_cloth_eval.py --ckpt outputs/lsim/cloth_$1/last.pt --tag $1 > outputs/lsim/real_eval_$1.log 2>&1
  CUDA_VISIBLE_DEVICES=$2 $L /tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/realcloth/model_whole.py outputs/lsim/cloth_$1/last.pt ${1}_pt > outputs/lsim/real_whole_$1.log 2>&1
  echo "[e chain] $1 evaluated $(date -u)"
}
ev v4e10 5 &
ev v4e100 5 &
(while kill -0 $1 2>/dev/null; do sleep 60; done; ev v4e1000 4) &
wait
echo "[e chain] all done $(date -u)"
