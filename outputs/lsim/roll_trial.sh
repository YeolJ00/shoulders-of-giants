#!/bin/bash
# The long-rollout trial (2026-11-25). The pushforward trial (pf_trial.sh) asked whether practising longer on the
# network's own output stops cloth's blow-up tail; its first arms say no: pushforward 8 put one frame +0.005 and
# blow-ups 31 against the control's 25 of 336, and rigid pushforward 16 left replays level. Its READ names the next
# step: train on the states the network's own long rollouts reach, with the teacher's trajectory as the target -- the
# long-rollout loss already in the trainer (cloth_train --roll: n episodes a step rolled out alone R frames from a
# random start with true history, mean distance of every free vertex from the teacher's every frame, the graph cut
# every 10 frames; lsim.train.cloth_real.rollout_loss). In October it took the 64M world-frame network from 27.8 /
# 71.7 / 104.3 mm to 25.4 / 65.2 / 86.1 at 1 s / 2 s / 3.8 s in 500 steps (q64L, lr 3e-5, 100 frames). Never tried on
# the own-frame networks. Smoke-tested 2026-11-25 (R 60, n 2, one GPU): runs, ~40 s a step, 10 GB.
#
#   all from cloth_q64_ownup_cont2/step23999.pt, every input given (--given, and --roll_given 1: the material given in
#   the rollouts too), the one-step loss as trained (K 8, pf 4), 400 steps, lr 3e-5 (v6e / q64L's), one GPU each:
#     cloth_roll_ctl   no rollout loss (the control; ~15 min)
#     cloth_roll60     --roll 60  (2 s)
#     cloth_roll100    --roll 100 (3.3 s, near the 3.8 s scoring horizon where blow-ups are counted)
#   scored at steps 200 and 399 on the 336 held-out episodes with the material given (l2_coverage_score --roll_given):
#   one frame, rollouts at 1 s / 2 s / 3.8 s, strain, area, blow-ups (area > 2x at the end).
#
# PREDICTION (written before the run):
#   control: one frame within 0.01 of cont2 (0.486) and blow-ups 20-30 (the pushforward control had 25).
#   roll60 / roll100: one frame within +0.02 of the control; 1 s level (within 1 mm); 3.8 s closer by 3-10 mm;
#   blow-ups at most two thirds of the control's (roll60) and at most half (roll100); strain mean lower.
# READ: blow-ups halved at a one-frame cost under 0.02 = the long-rollout loss joins the recipe (a longer fine-tune of
# the reference network with it next). Rollouts closer but the tail unchanged = distance to a chaotic teacher does not
# reach the instability; next is to look at where blow-ups start (render their first second) before any new loss.
# Neither = the instability is not in states 2-3 s from a true one.
#   usage: roll_trial.sh <gpu for control then roll100> <gpu for roll60> <gpu for the control's scoring>
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
wait_gpu() {
  local u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1==g{print $2}')
  busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
  while busy || { sleep 60; busy; }; do sleep 60; done
}
train() {   # name gpu extra-args...
  local n=$1 g=$2; shift 2
  echo "[roll] $n on GPU $g $(date -u)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --data data/datasets/c1 \
    --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --batch 4 --sig 1e-3 --prob_slots --unc --accel \
    --layer 2 --edge 1.0 --side_apart --own_up --given --pf 4 --steps 400 --lr 3e-5 --save_every 200 --keep \
    --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64_ownup_cont2/step23999.pt \
    --out outputs/lsim/$n "$@" > outputs/lsim/$n.log 2>&1
  echo "[roll] $n trained ($?) $(date -u)"
  rm -f outputs/lsim/$n/step0.pt
}
score() {   # name gpu
  local n=$1 g=$2
  for s in 200 399; do
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step$s.pt --tag cov_${n}_step${s}_rg \
      --roll_given > outputs/lsim/coverage_score_${n}_step$s.log 2>&1 &
  done
  wait
  echo "[roll] $n scored $(date -u)"
}
A=$1; B=$2; C=$3
{ wait_gpu $A; train cloth_roll_ctl $A; { wait_gpu $C; score cloth_roll_ctl $C; } &
  train cloth_roll100 $A --roll 100 --roll_given 1 --roll_n 2 --roll_every 10; score cloth_roll100 $A; wait; } &
{ wait_gpu $B; train cloth_roll60 $B --roll 60 --roll_given 1 --roll_n 2 --roll_every 10; score cloth_roll60 $B; } &
wait
echo "[roll] done $(date -u)"
