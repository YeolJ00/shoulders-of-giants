#!/bin/bash
# v6e (lr 3e-5), v6f (lr 1e-4, roll_w 0.1), v6g (v6e + 30% PGND-window teacher episodes): 100-frame rollout fine-tunes of v3
# after v6a/v6b (lr 3e-4) eroded the one-step fit (held-out clean step 0.59 -> 2.36 mm by step 1,000). Scored as v6_chain.sh, on GPU 4.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
for s in 500 1000 1500; do
  for r in v6e v6f v6g; do
    until [ -f outputs/lsim/cloth_$r/step$s.pt ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
    CUDA_VISIBLE_DEVICES=4 $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
    CUDA_VISIBLE_DEVICES=4 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_$r/step$s.pt --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
    echo "[v6efg chain] $r step $s evaluated $(date -u)"
  done
done
