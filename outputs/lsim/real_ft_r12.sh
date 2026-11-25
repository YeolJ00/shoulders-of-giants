#!/bin/bash
# r12: r5's real fine-tune from v6e2 step 1000 (the best imitation so far: c1_test 29 / 76 / 100 mm) -- does a better
# imitation lower the real plateau (~25 mm at 4 s)? GPU 0, scored at 500 and 1000.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
I=outputs/lsim/cloth_v6e2/step1000.pt
go(){ r=$1; g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $L -u scripts/l2_real_cloth_finetune.py --init $I --out outputs/lsim/cloth_$r --back_w 1 --strain_w 1 "$@" > outputs/lsim/cloth_$r.log 2>&1 &
  for s in 500 1000; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || return
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[r12] $r step $s evaluated $(date -u)"
  done; }
go r12 0 & wait
