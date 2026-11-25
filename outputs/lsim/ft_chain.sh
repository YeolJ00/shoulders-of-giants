#!/bin/bash
# 2026-10-22: when the v4r / v4c fine-tunes (GPU 3) end, evaluate each there, one at a time.
#   nohup outputs/lsim/ft_chain.sh <v4r pid> <v4c pid> &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
for pair in "v4r $1" "v4c $2"; do
  set -- $pair
  while kill -0 $2 2>/dev/null; do sleep 60; done
  grep -q Traceback outputs/lsim/cloth_$1.log && echo "[ft chain] $1 failed $(date -u)"
  echo "[ft chain] $1 ended $(date -u)"
  outputs/lsim/eval_cloth_ft.sh outputs/lsim/cloth_$1/last.pt $1 3
done
echo "[ft chain] all done $(date -u)"
