#!/bin/bash
# The two-sided real fine-tunes cloth_r2 (back_w 0.5) and cloth_r3 (back_w 1): each checkpoint on PGND's held-out
# recordings with the score's blind side reported (back, strain, area), on GPU 6.   nohup outputs/lsim/real_ft_chain2.sh &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for s in 250 500 750 1000; do
  for r in r2 r3; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
    CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[real ft 2] $r step $s evaluated $(date -u)"
  done
done
