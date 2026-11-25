#!/bin/bash
# Claim 3 with the local-frame network (2026-11-23). cloth_q64_ownup (all of c1, one frame 0.588, held-out rollouts
# 17.6 / 64.7 / 97.0 mm, strain 12%) on PGND's held-out windows BEFORE any real data (l2_real_cloth_pgnd_q64_ownup):
# 31.2 / 44.9 mm at 2 s / 4 s (the layer alone 33.2 / 39.4; the world-frame q64 25.9 / 50.8 with the cloth shredded,
# area 8.4x; its side-apart fine-tune 22.0 / 31.9), strain 4.5% / 6.2%, area 1.0x / 1.2x -- it keeps real cloth's shape
# and is no closer to it. Here: r5's real fine-tune exactly as it was run for q64 (rq64: two-sided + strain, 1,000
# steps, lr 1e-4, gradients through 10 frames, teacher loss weight 1; rq64 reached 17.7 / 18.2 mm with strain 7-8%,
# the fitted teacher 15.8 / 16.8), scored at steps 500 and 1,000.
# PREDICTION (written before the run): level with rq64 at step 1,000 (16.2-19.2 mm at 2 s, 16.7-19.7 at 4 s) with less
# stretch (6% or under): the real plateau moved with capacity before, not with the teacher fit, and before real data
# this network is further from real cloth than the world-frame one. Wrong on the good side if it reaches the fitted
# teacher's 15.8 / 16.8 (claim 3); wrong on the bad side if it stays above 20 (then the local frames transfer worse
# to PGND's two-gripper, 50-63 cm cloths than world frames do).
#   $1 = GPU (waits until it has no compute app twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[realft] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L -u scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_q64_ownup/step23999.pt \
  --out outputs/lsim/cloth_rq64_ownup --steps 1000 --lr 1e-4 --trunc 10 --teacher_w 1 --back_w 1 --strain_w 1 \
  > outputs/lsim/cloth_rq64_ownup.log 2>&1
echo "[realft] trained ($?) $(date -u)"
for s in 500 1000; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_rq64_ownup/step$s.pt --tag rq64_ownup_step$s \
    > outputs/lsim/pgnd_eval_rq64_ownup_step$s.log 2>&1 &
done
wait
echo "[realft] scored $(date -u)"
