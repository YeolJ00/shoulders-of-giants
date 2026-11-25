#!/bin/bash
# The real-data fine-tunes cloth_r0 (real only) and cloth_r1 (+ the teacher loss): each checkpoint, as it appears,
# on PGND's held-out recordings (scripts/l2_real_cloth_pgnd_eval.py, GPU 3).   nohup outputs/lsim/real_ft_chain.sh &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for s in 250 500 750 1000; do
  for r in r0 r1; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
    CUDA_VISIBLE_DEVICES=3 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1 &
  done
  wait
  echo "[real ft] step $s evaluated $(date -u)"
done
