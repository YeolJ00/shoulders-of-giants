#!/bin/bash
# The held-out comparison for --own_up (2026-11-22): the coverage study's "seven" arm (2,000 episodes of all seven
# gripper families, 12,000 steps x 1 GPU, q64's recipe + --side_apart) with --own_up, against its world-frame twin
# cloth_cov_seven (one frame on c1_test, error / constant velocity: fold 0.87, shake 0.84, the other five 0.79-1.05;
# the trainer's held-out val 1.87 / 1.50 / 1.47 / 0.86 / 0.78 / 0.72 mm at steps 500 / 1,000 / 3,000 / 6,000 / 9,000 /
# 11,999). The 16-episode run with --own_up reads 2.20 mm on held-out cloth at step 500 (world frame 2.46, --own
# alone 4.63). GPU $1 once it is empty twice a minute apart; scored like every arm by l2_coverage_score.py.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[ownup7] GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --subset outputs/lsim/coverage_subsets/seven.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/cloth_cov_seven_ownup > outputs/lsim/cloth_cov_seven_ownup.log 2>&1
echo "[ownup7] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven_ownup/step11999.pt \
  --tag cov_seven_ownup_step11999 > outputs/lsim/coverage_score_seven_ownup.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_cov_seven_ownup/step9000.pt --n_roll 0 \
  --tag cov_seven_ownup_step9000 > outputs/lsim/coverage_score_seven_ownup_9000.log 2>&1 &
wait
echo "[ownup7] scored $(date -u)"
