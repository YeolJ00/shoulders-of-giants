#!/bin/bash
# v6d (v6b + 30% teacher episodes on PGND's windows: the coverage arm) checkpoints as they appear, scored as v6_chain.sh, on GPU 6.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for s in 500 1000 1500; do
  r=v6d
  until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
  [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
  echo "[v6d chain] $r step $s evaluated $(date -u)"
done
