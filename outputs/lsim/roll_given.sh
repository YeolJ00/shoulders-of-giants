#!/bin/bash
# The long-rollout loss on the reference cloth network (2026-11-25). The long-rollout trial (roll_trial.sh), from the
# third stretch trained with the material hidden, 400 steps, lr 3e-5, everything given: the 3.3 s arm (--roll 100)
# halved its blow-ups (14 of 336 against the control's 30; 20 against 32 at step 200), rollouts closer at 2 s and
# 3.8 s (-4.5 [-6.9, -2.5] / -6.2 [-9.0, -3.6] mm), one frame +0.005 [+0.002, +0.008], strain mean 26 -> 9.5% but the
# median up 1.0 -> 1.8% (most cloths a little more stretched, few blown up); the 2 s arm (--roll 60) left blow-ups
# level. Against the reference cloth network (the third stretch with every input given,
# cloth_q64_ownup_cont2_given/step23999.pt: one frame 0.475, 13.7 / 50.4 / 76.7 mm, strain 4.2% (1.1%), 5 blow-ups)
# that arm is closer over seconds (-3.6 [-7.3, +0.8] / -4.8 [-10.8, +1.1]) but worse on one frame (+0.016), stretch
# (9.5%) and blow-ups (14). Its READ: blow-ups halved at under 0.02 = a longer fine-tune of the reference with it.
#   both from cloth_q64_ownup_cont2_given/step23999.pt, --given, the one-step loss as trained (K 8, pf 4), lr 3e-5:
#     cloth_roll_ctl_given   800 steps, no rollout loss (the control; ~40 min, beside the other on the same GPU)
#     cloth_roll100_given    800 steps, --roll 100 --roll_given 1 --roll_n 2 --roll_every 10 (~15 h)
#   scored on the 336 held-out episodes with the material given (l2_coverage_score --roll_given): the control at
#   799, the rollout arm at 400 (while it trains) and 799.
# PREDICTION (written before the run):
#   control: one frame within 0.005 of the reference, 3-8 blow-ups, positions within 1.5 mm.
#   rollout arm (both checkpoints): one frame +0.003 to +0.010; level at 1 s (0.5 mm); closer by 2-6 mm at 2 s and
#   3-8 mm at 3.8 s; at most 5 blow-ups; strain mean at most 6%, median at most 1.8%.
# READ: closer over seconds with no more blow-ups and at most 0.01 on one frame = the long-rollout loss becomes the
# last stage of cloth's teacher training (run next on the fourth stretch's network); level = from a network trained
# with every input given it adds nothing (it repaired the hidden-input tail only); more blow-ups or median strain
# above 2% = it costs shape on a sound network and stays a repair, not a stage.
#   usage: roll_given.sh <gpu>   (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
train() {   # name extra-args...
  local n=$1; shift
  echo "[rollg] $n on GPU $g $(date -u)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --data data/datasets/c1 \
    --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --batch 4 --sig 1e-3 --prob_slots --unc --accel \
    --layer 2 --edge 1.0 --side_apart --own_up --given --pf 4 --steps 800 --lr 3e-5 --save_every 400 --keep \
    --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64_ownup_cont2_given/step23999.pt \
    --out outputs/lsim/$n "$@" > outputs/lsim/$n.log 2>&1
  echo "[rollg] $n trained ($?) $(date -u)"
  rm -f outputs/lsim/$n/step0.pt
}
score() {   # name step
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$1/step$2.pt --tag cov_${1}_step${2}_rg \
    --roll_given > outputs/lsim/coverage_score_${1}_step$2.log 2>&1
  echo "[rollg] $1 step $2 scored $(date -u)"
}
{ train cloth_roll_ctl_given; score cloth_roll_ctl_given 799; } &
{ n=cloth_roll100_given
  train $n --roll 100 --roll_given 1 --roll_n 2 --roll_every 10 &
  until [ -f outputs/lsim/$n/step400.pt ] || ! kill -0 $! 2>/dev/null; do sleep 60; done
  [ -f outputs/lsim/$n/step400.pt ] && { sleep 60; score $n 400; }
  wait
  score $n 799; } &
wait
echo "[rollg] done $(date -u)"
