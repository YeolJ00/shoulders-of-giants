#!/bin/bash
# The compute lever for the local-frame network (2026-11-23). cloth_q64_ownup (all 20,032 c1 episodes, 24,000 steps x
# 2 GPUs x batch 4, --side_apart --own_up) ended at one frame 0.588 of constant velocity on c1_test (the world-frame
# q64: 0.688; q64 continued for another 24,000 steps: 0.667, i.e. -0.021 for twice the training) with rollouts 17.6 /
# 64.7 / 97.0 mm at 1 s / 2 s / 3.8 s (q64 continued: 24.4 / 79.6 / 121.7), strain 11.9%. Here: its step-23,999 weights
# + another 24,000 steps, new optimiser and schedule, exactly as cloth_q64_cont was made -- does the fit keep falling
# with twice the compute? To be set beside the data lever (cloth_q64_ownup_2k: 2,000 episodes at this run's compute).
# PREDICTION (written before the run): one frame ends 0.555-0.575 (-0.015 to -0.035: the world-frame network's gain or a
# little more, since this one was still falling 0.05 over its last third); rollouts within +-8 mm of 97 mm at 3.8 s.
# READ: a gain of 0.03 or more = compute is still a lever at this data size, comparable to what a decade of data is
# predicted to be worth; under 0.01 = a plateau, and what is left is the network or its inputs, not training length.
#   $1 = run name (outputs/lsim/<name>), $2 = "g1,g2" (two GPUs, each checked empty twice a minute apart)
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
echo "[cont] $n on GPUs $G $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29575 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --own_up \
  --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64_ownup/step23999.pt \
  --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[cont] $n trained ($?) $(date -u)"
g1=${G%%,*}; g2=${G##*,}
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --tag cov_${n}_step23999 \
  > outputs/lsim/coverage_score_${n}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g2 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_${n}_step23999_train > outputs/lsim/coverage_score_${n}_train.log 2>&1 &
wait
echo "[cont] $n scored $(date -u)"
