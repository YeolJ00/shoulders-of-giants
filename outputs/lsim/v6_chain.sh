#!/bin/bash
# v6a / v6b checkpoints, as they appear (every 500 steps): the long horizon on c1_test (scripts/l2_cloth_long.py)
# and PGND's held-out windows (scripts/l2_real_cloth_pgnd_eval.py), on GPU 6. v3's baseline first.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_v3/last.pt --tag v3 > outputs/lsim/long_v3.log 2>&1
for s in 500 1000 1500 2000 2500; do
  for r in v6a v6b; do
    [ $r = v6b ] && [ $s -gt 1500 ] && continue                      # v6b: 2,000 steps, its last save is last.pt
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
    CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
    CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[v6 chain] $r step $s evaluated $(date -u)"
  done
done
