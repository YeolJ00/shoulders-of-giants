#!/bin/bash
# The fourth 24,000 steps, every input given (2026-11-25). The compute lever so far on the full-data own-frame network
# (seven-family one frame / constant velocity on c1_test): 24k 0.588, 48k 0.520 (-0.068), 72k 0.486 hidden-input
# recipe (-0.034) or 0.475 with every input given (cloth_q64_ownup_cont2_given, -0.045), no train / held-out gap at
# any stage (72k given: its own episodes 0.489). With every input given the third stretch also kept the cloth:
# blow-ups 5 against 28 of 336, strain mean 4.2% against 24.3%, rollouts 13.7 / 50.4 / 76.7 mm at 1 s / 2 s / 3.8 s.
# It is the reference cloth network. Here: another 24,000 steps from it, the same recipe and schedule (OneCycle,
# lr 6e-4, 2 GPUs x batch 4):
#   cloth_q64_ownup_cont3_given   --init outputs/lsim/cloth_q64_ownup_cont2_given/step23999.pt --given
# scored with the material given (--roll_given) on the 336 held-out episodes, and on 2,000 of its training episodes.
# PREDICTION (written before the run): one frame 0.450-0.465 (the gain halving again: -0.01 to -0.025); rollouts
# 13-14 / 46-50 / 70-77 mm; blow-ups at most 8; strain mean at most 6%; no train / held-out gap (within 0.02).
# READ: a gain of 0.015 or more = training length is still the lever; the reference network moves here. Below 0.01 =
# the fit is flattening at this size and data -- then the size question returns (509M was level with 64M at equal
# windows seen, 2026-11-20, but at a third of today's training) and, if a gap opens, the data question. A tail that
# grows back = the tail grows with training whatever the inputs, and the given recipe only delayed it.
#   $1 = "g1,g2" (two GPUs, each empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
T=/home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun
n=cloth_q64_ownup_cont3_given; G=$1
busy() {
  for g in ${G//,/ }; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 0
  done
  return 1
}
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[given3] $n on GPUs $G $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29578 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --own_up \
  --given --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" \
  --init outputs/lsim/cloth_q64_ownup_cont2_given/step23999.pt --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[given3] $n trained ($?) $(date -u)"
g1=${G%%,*}; g2=${G##*,}
for s in 20000 23999; do
  CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step$s.pt --tag cov_${n}_step${s}_rg \
    --roll_given > outputs/lsim/coverage_score_${n}_step$s.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=$g2 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_${n}_step23999_train > outputs/lsim/coverage_score_${n}_train.log 2>&1 &
wait
echo "[given3] $n scored $(date -u)"
