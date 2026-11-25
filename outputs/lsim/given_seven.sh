#!/bin/bash
# Cloth with ALL INPUTS GIVEN (2026-11-24; the user: "maybe we should start with all inputs given"). On rigid the
# hidden inputs cost the plain transformer a third of its fit (0.571 -> 0.765 on held-out episodes, the random yaw
# none of it). The cloth recipe hides too: each parameter-slot group with probability 0.5, the grasp force in 30% of
# windows, a random shorter past. Here cloth_cov_seven_ownup's exact twin (2,000 episodes of all seven families,
# 12,000 steps x 1 GPU x batch 4, --own_up, the noise and pushforward, the random yaw kept) + --given (cloth_train:
# p_hidden 0, p_hide_grasp 0, all 8 past frames; scored as always with everything given):
#   cloth_cov_seven_ownup_given     against cloth_cov_seven_ownup (one frame 0.949 / 0.866 / 0.780 / 0.753 at 3,000 /
#   6,000 / 9,000 / 11,999; rollouts on 336 held-out episodes 25.6 / 78.8 / 115.4 mm)
# PREDICTION (written before the run): little -- within 0.02 of the twin on one frame at the end (0.73-0.77) and
# rollouts level. Reasons: at one material the cloth network fit no better (0.628 against 0.632), it ignores a
# given material at test time (given = hidden within 0.3 mm), and its slot loss is already off the trunk
# (--side_apart). READ: ahead by >= 0.05 = the hidden inputs cost cloth as they cost rigid, and the fit study's
# 0.59 was partly the price of hiding; within 0.02 = for cloth the hiding is free and can stay or go.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=cloth_cov_seven_ownup_given
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[given7] GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up --given --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --subset outputs/lsim/coverage_subsets/seven.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[given7] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step11999.pt \
  --tag cov_seven_ownup_given_step11999 > outputs/lsim/coverage_score_seven_ownup_given.log 2>&1 &
for s in 3000 6000 9000; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step$s.pt --n_roll 0 \
    --tag cov_seven_ownup_given_step$s > outputs/lsim/coverage_score_seven_ownup_given_$s.log 2>&1 &
done
wait
echo "[given7] scored $(date -u)"
