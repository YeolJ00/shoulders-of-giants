#!/bin/bash
# Claim 3 with the best cloth network (2026-11-24). r5's real fine-tune, unchanged, from cloth_q64_ownup_cont/
# step23999.pt (local frame, twice the training: one frame 0.520 on c1_test; on PGND before real data 25.6 / 34.7 mm
# at 2 s / 4 s, 5.6 [2.2, 9.5] / 10.2 [2.0, 19.0] closer than its 24k self). The 24k network after this fine-tune
# (rq64_ownup): 17.6 / 19.8 mm, level with the world-frame rq64 (17.7 / 18.2); fitted teacher 15.8 / 16.8.
#   cloth_rq64_ownup_cont
# PREDICTION (written before the run): level again, 16.5-18.5 / 17.5-20.0 mm, strain under 3% -- two networks of
# different teacher fit both stopped at ~18 mm, so the real plateau is set by the real fine-tune (its data, its
# 10-frame gradient window), not by the teacher fit. READ: below 16.5 at 2 s = the better teacher fit carries
# through real fine-tuning, and claim 3 is a matter of teacher fit; below 15.8 = claim 3 shown on PGND.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[realft2] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L -u scripts/l2_real_cloth_finetune.py --init outputs/lsim/cloth_q64_ownup_cont/step23999.pt \
  --out outputs/lsim/cloth_rq64_ownup_cont --steps 1000 --lr 1e-4 --trunc 10 --teacher_w 1 --back_w 1 --strain_w 1 \
  > outputs/lsim/cloth_rq64_ownup_cont.log 2>&1
echo "[realft2] trained ($?) $(date -u)"
for s in 500 1000; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_rq64_ownup_cont/step$s.pt --tag rq64_ownup_cont_step$s \
    > outputs/lsim/pgnd_eval_rq64_ownup_cont_step$s.log 2>&1 &
done
wait
echo "[realft2] scored $(date -u)"
