#!/bin/bash
# Three hypotheses about v6e's plateau (c1_test 30 / 85 / 107 mm at 1 / 2 / 3.8 s; still 57 / 144 / 168, the layer
# alone 70 / 155 / 176), each on its own GPU, 2,000 steps, scored at 500 / 1000 / 1500 / 1999 on GPU 7:
#   v6j  v6e continued at a 1e-5 peak (v6e2 / v6i at 3e-5 lost ground in their re-warm-up)          GPU 0
#   v6k  from v3, v6e's recipe with the rollout weight 3                                             GPU 1
#   v6l  from v3, 50-frame rollouts, 4 a step (cheaper frames, more episodes per gradient)            GPU 2
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
tr(){ r=$1; g=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g $L -u -m lsim.train.cloth_train --data data/datasets/c1 --steps 2000 --save_every 500 --keep --batch 4 \
    --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 --roll_every 10 "$@" \
    --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$r > outputs/lsim/cloth_$r.log 2>&1 & }
tr v6j 0 --init outputs/lsim/cloth_v6e/step1999.pt --lr 1e-5 --roll 100 --roll_n 2 --roll_w 1
tr v6k 1 --init outputs/lsim/cloth_v3/last.pt --lr 3e-5 --roll 100 --roll_n 2 --roll_w 3
tr v6l 2 --init outputs/lsim/cloth_v3/last.pt --lr 3e-5 --roll 50 --roll_n 4 --roll_w 1
ev(){ r=$1; s=$2; f=outputs/lsim/cloth_$r/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_$r.log; do sleep 60; done
  [ -f $f ] || return; sleep 30
  CUDA_VISIBLE_DEVICES=7 $L scripts/l2_cloth_long.py --ckpt $f --tag ${r}_step$s > outputs/lsim/long_${r}_step$s.log 2>&1
  CUDA_VISIBLE_DEVICES=7 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag ${r}_step$s > outputs/lsim/pgnd_eval_${r}_step$s.log 2>&1
  echo "[v6jkl] $r step $s evaluated $(date -u)"; }
for s in 500 1000 1500 1999; do for r in v6l v6j v6k; do ev $r $s; done; done
