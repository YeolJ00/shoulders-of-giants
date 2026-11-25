#!/bin/bash
# The cloth compute lever, third segment (2026-11-24). full_ownup_cont.sh's result: the local-frame network trained
# for a second 24,000 steps (cloth_q64_ownup_cont) scores one frame 0.520 against 0.588 on c1_test (-0.068 [-0.078,
# -0.058], every family), 0.539 on 2,000 of its training episodes (no gap), rollouts on 336 held-out episodes 15.0 /
# 57.2 / 85.1 mm against 17.6 / 64.7 / 97.0 (-2.6 [-3.5, -1.7] / -7.4 [-10.8, -4.3] / -11.9 [-19.0, -5.7]), strain
# 6.1% against 11.9% (median 1.3% against 3.4%). The prediction (0.555-0.575; rollouts within 8 mm of 97) was too
# timid on both. Twice the training is worth 0.068 where ten times the data was worth 0.015 (and where the
# world-frame network gained 0.021 from the same doubling): training length is THE lever for this network. Here a
# third 24,000 steps from that checkpoint, a new optimiser and schedule as before, recipe unchanged (hidden inputs
# kept, so that the segment is comparable; the all-inputs-given test is cloth_cov_seven_ownup_given):
#   cloth_q64_ownup_cont2      --init outputs/lsim/cloth_q64_ownup_cont/step23999.pt
# PREDICTION (written before the run): one frame ends 0.475-0.500 (a further -0.02 to -0.045: 72k over 48k steps is
# half a doubling), still no train / held-out gap, rollouts 76-84 mm at 3.8 s. READ: a gain of 0.02 or more = keep
# training (the fit is bought with steps); under 0.01 = the plateau, and what is left is the network or its inputs.
#   $1 = run name, $2 = "g1,g2" (two GPUs, each empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
T=/home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun
n=$1; G=$2; shift 2
busy() {
  for g in ${G//,/ }; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 0
  done
  return 1
}
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[cont2] $n on GPUs $G $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29576 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --own_up \
  --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64_ownup_cont/step23999.pt \
  --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[cont2] $n trained ($?) $(date -u)"
g1=${G%%,*}; g2=${G##*,}
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --tag cov_${n}_step23999 \
  > outputs/lsim/coverage_score_${n}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g2 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_${n}_step23999_train > outputs/lsim/coverage_score_${n}_train.log 2>&1 &
wait
echo "[cont2] $n scored $(date -u)"
