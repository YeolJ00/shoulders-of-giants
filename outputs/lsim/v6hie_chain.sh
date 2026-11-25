#!/bin/bash
# Pending checkpoints of v6g (final), v6h (slots given half the rollouts), v6i (6 rollouts a step) and v6e2 (v6e
# continued), as they appear: c1_test long horizon and PGND held-out, on GPU 6. Order = expected arrival.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
ev(){ r=$1; s=$2; f=outputs/lsim/cloth_$r/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
  [ -f $f ] || return; sleep 30
  [ -f docs/results/l2_real_cloth_pgnd_${r}_step$s.json ] && return
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt $f --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
  echo "[v6hie] $r step $s evaluated $(date -u)"; }
ev v6g 1999; ev v6h 500; ev v6i 500; ev v6e2 1000; ev v6h 1000; ev v6i 1000; ev v6h 1500; ev v6i 1499; ev v6e2 2000
ev v6h 1999; ev v6e2 3000; ev v6e2 3999
