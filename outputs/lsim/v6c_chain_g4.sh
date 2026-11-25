#!/bin/bash
# v6c (v6a + 30% teacher episodes on PGND's windows: the coverage arm) checkpoints as they appear, scored as v6_chain.sh, on GPU 4 (moved from 6: another user took it).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for s in 500 1000 1500 2000 2500; do
  r=v6c
  until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
  [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
  [ -f docs/results/l2_real_cloth_pgnd_${r}_step$s.json ] && continue
  CUDA_VISIBLE_DEVICES=4 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
  CUDA_VISIBLE_DEVICES=4 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
  echo "[v6c chain] $r step $s evaluated $(date -u)"
done
