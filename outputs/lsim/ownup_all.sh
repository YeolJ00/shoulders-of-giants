#!/bin/bash
# Does more data help the candidate network at EQUAL compute? (2026-11-23) --own_up on all 20,032 c1 episodes with the
# 2,000-episode run's recipe and compute exactly (cloth_cov_seven_ownup: 12,000 steps x 1 GPU x batch 4 = 48k windows,
# the same cosine schedule), so the three runs 200 / 2,000 / 20,032 episodes differ in the data alone (each window
# seen 2.3 / 0.23 / 0.023 times). 2,000 episodes ended at one frame 0.753 on c1_test (0.871 world frame) and rollouts
# 25.6 / 78.8 / 115.4 mm at 1 s / 2 s / 3.8 s (336 held-out episodes).
# PREDICTION (written before the run; DIRECTION.md): within +-0.02 of 0.753 on one frame and within the +-10 mm
# checkpoint swing on rollouts -- at this compute every window is new to both runs, so 10x the episodes adds nothing;
# the lever is compute. If instead it is clearly lower (< 0.72), data diversity matters even at fixed compute and the
# "scale beyond 20k?" answer has to be re-opened.
#   cloth_cov_all_ownup   GPU $1 once it is empty twice a minute apart; scored like every arm by l2_coverage_score.py
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[ownall] GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/cloth_cov_all_ownup > outputs/lsim/cloth_cov_all_ownup.log 2>&1
echo "[ownall] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_all_ownup/step11999.pt \
  --tag cov_all_ownup_step11999 > outputs/lsim/coverage_score_all_ownup.log 2>&1
echo "[ownall] scored ($?) $(date -u)"
