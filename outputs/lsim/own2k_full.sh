#!/bin/bash
# What is data worth at FULL compute? (2026-11-23) --own_up on the 2,000-episode subset (outputs/lsim/coverage_subsets/
# seven.json) with the full-data run's recipe and compute exactly (cloth_q64_ownup: 24,000 steps x 2 GPUs x batch 4 =
# 192k windows, the same schedule), so the two runs differ in the data alone: 2,000 against 20,032 episodes.
# What is known: at the small budget (12,000 steps x 1 GPU = 48k windows) data size does nothing on one frame --
# 200 / 2,000 / 20,032 episodes: 0.781 / 0.753 / 0.754 on c1_test (all-data against 2,000: +0.001 [-0.007, +0.012]),
# train / held-out gap 0.06 / 0.017 / -- ; rollouts 114.5 / 115.4 / 133.9 mm at 3.8 s. The full-data run at full
# compute is at 0.638 at step 16,000 (q64 0.731) and still annealing.
# PREDICTION (written before the run; DIRECTION.md): here each of the subset's ~212k windows is seen ~0.9 times
# (200 episodes at the small budget: 2.3 times, gap 0.06; 2,000 at the small budget: 0.23 times, gap 0.017), so a gap
# of 0.03-0.05 opens and the run ends 0.02-0.05 above the full-data run on held-out episodes, level with it on its own
# episodes. DECISION RULE for "scale beyond 20k?": the held-out difference between 2,000 and 20,032 episodes at this
# compute is what a decade of data is worth here. < 0.01: not indicated at this compute. >= 0.03: a decade is worth
# about that, and 200k episodes are worth testing once the compute scales with them.
#   $1 = run name, $2 = "g1,g2" (two GPUs, each checked empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
T=/home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun
n=$1; G=$2
busy() {
  for g in ${G//,/ }; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 0
  done
  return 1
}
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[own2k] $n on GPUs $G $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29573 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --own_up \
  --subset outputs/lsim/coverage_subsets/seven.json \
  --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[own2k] $n trained ($?) $(date -u)"
g1=${G%%,*}; g2=${G##*,}
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --tag cov_${n}_step23999 \
  > outputs/lsim/coverage_score_${n}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g2 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_${n}_step23999_train > outputs/lsim/coverage_score_${n}_train.log 2>&1 &
wait
echo "[own2k] $n scored $(date -u)"
