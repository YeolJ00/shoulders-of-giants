#!/bin/bash
# r10 / r11: r5's recipe from v6e step 1000 with gradients through 20 / 30 frames before each observed cloud (--trunc;
# r4-r9 used 10 and all plateau at 25-31 mm at 4 s: PGND's clouds are 60 frames apart, so 80 of the 113 rolled-out
# frames never get a gradient). GPUs 2 and 7, each scored at 500 and 1000.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
I=outputs/lsim/cloth_v6e/step1000.pt
go(){ r=$1; g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $L -u scripts/l2_real_cloth_finetune.py --init $I --out outputs/lsim/cloth_$r --back_w 1 --strain_w 1 "$@" > outputs/lsim/cloth_$r.log 2>&1 &
  for s in 500 1000; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || return
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[r1011] $r step $s evaluated $(date -u)"
  done; }
go r10 2 --trunc 20 & go r11 7 --trunc 30 & wait
