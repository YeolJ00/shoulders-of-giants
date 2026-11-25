#!/bin/bash
# The Qwen-style runs' checkpoints (every 2,000 steps) on the long horizon (scripts/l2_cloth_long.py: c1_test from
# frame 7, all slots hidden; compare paired against v3 / v6e on the same 64 episodes), on GPU 6 beside q64.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
ev(){ r=$1; s=$2; f=outputs/lsim/cloth_$r/step$s.pt
  until [ -f $f ] || grep -q "Error" outputs/lsim/cloth_$r.log; do sleep 120; done
  [ -f $f ] || return; sleep 60
  [ -f docs/results/l2_cloth_long_${r}_step$s.json ] && return
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt $f --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
  echo "[q] $r step $s evaluated $(date -u)"; }
(for s in $(seq 2000 2000 22000) 23999; do ev q64 $s; done) &
(for s in $(seq 2000 2000 10000) 11999; do ev q509 $s; done) &
wait
