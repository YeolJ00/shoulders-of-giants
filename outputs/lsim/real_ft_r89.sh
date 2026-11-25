#!/bin/bash
# r8: r5's recipe from v6e step 1000 with no teacher loss (--teacher_w 0; back + strain guard it), GPU 1.
# r9: the same with the teacher loss, windows drawn ~ motion^1 (the gap to the teacher sits on the moving half), GPU 3.
# Each scored at 500 and 1000 on PGND held-out, on its own GPU.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
I=outputs/lsim/cloth_v6e/step1000.pt
go(){ r=$1; g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $L -u scripts/l2_real_cloth_finetune.py --init $I --out outputs/lsim/cloth_$r --back_w 1 --strain_w 1 "$@" > outputs/lsim/cloth_$r.log 2>&1 &
  for s in 500 1000; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || return
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[r89] $r step $s evaluated $(date -u)"
  done; }
go r8 1 --teacher_w 0 & go r9 3 --motion_pow 1 & wait
