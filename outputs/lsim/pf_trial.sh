#!/bin/bash
# The pushforward trial (2026-11-25, the user: "yes, set up the pushforward trial"). On both tracks longer training
# still improves one frame but no longer how the network runs alone: cloth's third 24k (cloth_q64_ownup_cont2) grew
# a blow-up tail (21-27 of 336 held-out rollouts above 2x area against 6-12), and the rigid plain transformer at 9x
# the training (rigid_pilot1024p_attn_direct_108k: one frame 0.368) lets bodies creep after contact (moved bodies end
# +8.3 [+4.1, +13.8] mm farther than at 36k). Both trainers practise running alone only through the pushforward:
# the first r of the 8 chunk frames are the network's own rollout (r uniform in 1..pf; cloth pf 4, rigid pf 6), so
# they never see their own state more than 4-6 frames (0.13-0.2 s) after a true one. Here: the same trained network,
# fine-tuned briefly three ways that differ only in how far it runs alone in training (--K = chunk frames per window,
# new option; --pf up to K). Short fine-tunes, paired against a control fine-tune of the same length and lr.
#
#   cloth, from cloth_q64_ownup_cont2/step23999.pt, every input given (--given), 3,000 steps, lr 1e-4, one GPU each:
#     cloth_pf_ctl   --K 8  --pf 4    (as trained)
#     cloth_pf8      --K 8  --pf 8    (the whole current window)
#     cloth_pf16     --K 16 --pf 16   (half a second)
#   scored at steps 2,000 and 2,999 on the 336 held-out episodes with the material given (l2_coverage_score
#   --roll_given): one frame, rollouts at 1 s / 2 s / 3.8 s, and the blow-up count (area > 2x at the end).
#
#   rigid, from rigid_pilot1024p_attn_direct_108k.pt (1,024 episodes, --plain as trained), 6,000 steps, lr 2e-4:
#     rigid_pf_ctl   --K 8  --pf 6    (as trained)
#     rigid_pf16     --K 16 --pf 16
#     rigid_pf30     --K 30 --pf 30   (one second)
#   scored at steps 4,000 and 6,000 on 32 unseen episodes of shards 8 and 1 (l2_small_fit_rigid: one frame and the
#   replay; paired over moved bodies by scratch rigid_replay_paired.py).
#
# PREDICTION (written before the run):
#   cloth: the control within 0.01 of cont2 on one frame (0.486) with a tail like cont2's scored the same way;
#   pf8 one frame within +0.02 of the control and fewer blow-ups (at most two thirds of the control's); pf16 one frame
#   +0.01 to +0.04 behind the control, blow-ups at most half the control's, 2 s level (within 4 mm).
#   rigid: the control within 0.02 of 0.368; pf16 and pf30 one frame within +0.03 of the control, moved bodies'
#   replay end closer than the control's by 5 mm or more (pf30 more than pf16).
# READ: fewer blow-ups / less creep at a one-frame cost under 0.02 = training on longer own rollouts is the next
# long run's recipe (and the step-count lever is not exhausted, only mis-aimed); no change = the instability is not
# about the first half-second of the network's own error -- next is states from its own long rollouts labelled by the
# teacher (restart the teacher there) or the long-rollout loss.
#   $1 = track (cloth | rigid), then one GPU per arm, e.g.  pf_trial.sh cloth 0 1 4   /   pf_trial.sh rigid 5 6 7
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
track=$1; shift
wait_gpu() {
  local u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1==g{print $2}')
  busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
  while busy || { sleep 60; busy; }; do sleep 60; done
}
cloth_arm() {   # name K pf gpu
  local n=$1 K=$2 pf=$3 g=$4
  wait_gpu $g; echo "[pf] $n on GPU $g $(date -u)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --data data/datasets/c1 \
    --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --batch 4 --sig 1e-3 --prob_slots --unc --accel \
    --layer 2 --edge 1.0 --side_apart --own_up --given --K $K --pf $pf --steps 3000 --lr 1e-4 --save_every 1000 --keep \
    --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64_ownup_cont2/step23999.pt \
    --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
  echo "[pf] $n trained ($?) $(date -u)"
  for s in 2000 2999; do
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step$s.pt --tag cov_${n}_step${s}_rg \
      --roll_given > outputs/lsim/coverage_score_${n}_step$s.log 2>&1 &
  done
  wait
  echo "[pf] $n scored $(date -u)"
}
rigid_arm() {   # name K pf gpu
  local n=$1 K=$2 pf=$3 g=$4
  wait_gpu $g; echo "[pf] $n on GPU $g $(date -u)"
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 6000 --lr 2e-4 \
    --pair_norm --coulomb --noise 0.003,0.015 --mix --plain --trunk attn --d 128 --layers 6 --direct --K $K --pf $pf \
    --init outputs/lsim/rigid_pilot1024p_attn_direct_108k.pt --save_every 2000 --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
  echo "[pf] $n trained ($?) $(date -u)"
  for c in $n ${n}_s4000; do
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$c.pt --episodes 32 --shard 8 --tag held > outputs/lsim/small_fit_${c}_held.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$c.pt --episodes 32 --shard 1 --tag shard1 > outputs/lsim/small_fit_${c}_shard1.log 2>&1 &
  done
  wait
  echo "[pf] $n scored $(date -u)"
}
if [ "$track" = cloth ]; then
  cloth_arm cloth_pf_ctl 8 4 $1 & cloth_arm cloth_pf8 8 8 $2 & cloth_arm cloth_pf16 16 16 $3 &
else
  rigid_arm rigid_pf_ctl 8 6 $1 & rigid_arm rigid_pf16 16 16 $2 & rigid_arm rigid_pf30 30 30 $3 &
fi
wait
echo "[pf] $track done $(date -u)"
