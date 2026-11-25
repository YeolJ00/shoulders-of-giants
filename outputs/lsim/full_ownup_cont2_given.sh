#!/bin/bash
# The cloth recipe at the frontier: every input given (2026-11-25). given_rescore.sh's result (2,000 episodes, 12,000
# steps, rollouts scored on the inputs each network trained with): the every-input-given network
# (cloth_cov_seven_ownup_given, material given in rollouts) against the usual hidden-input twin on the same 336
# held-out episodes: 1 s 23.8 against 25.6 mm (-1.8 [-3.3, -0.3]), 2 s level (-1.8 [-5.3, +1.7]), 3.8 s -8 [-17, +1],
# strain 12% against 61% (-49 [-55, -42]), area 1.06x against 2.23x (median 0.98 / 1.00); one frame 0.764 against
# 0.753. The random shorter past with everything else given keeps the stretch (strain 98%, area 3.7x). So cloth's
# recipe becomes --given (all 8 past frames). Here it is tested where the best network is: a third 24,000 steps from
# cloth_q64_ownup_cont/step23999.pt with --given, beside cloth_q64_ownup_cont2 (same init, same steps, same schedule,
# hidden-input recipe) -- the one difference is the recipe:
#   cloth_q64_ownup_cont2_given   --init outputs/lsim/cloth_q64_ownup_cont/step23999.pt --given
# Both scored with the material given in rollouts (--roll_given; the hidden-input network ignores a given material,
# twin within 0.3 mm at 1 s).
# PREDICTION (written before the run): one frame within 0.015 of cont2 (the twin pair: +0.012); rollouts within 3 mm
# at 1 s and 8 mm at 2 s; strain mean at most half of cont2's (the 48k network: 6.1%, median 1.3%) and area mean
# closer to 1. READ: lower stretch at no cost in position = --given is the cloth recipe from here on (and the
# reference network for the hand-over); position worse by more than 3 mm at 1 s = the hidden-input training was doing
# something for long rollouts, keep the old recipe and say so.
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
echo "[given] $n on GPUs $G $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29577 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart --own_up \
  --given --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" \
  --init outputs/lsim/cloth_q64_ownup_cont/step23999.pt --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[given] $n trained ($?) $(date -u)"
g1=${G%%,*}; g2=${G##*,}
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --tag cov_${n}_step23999_rg \
  --roll_given > outputs/lsim/coverage_score_${n}.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_q64_ownup_cont2/step23999.pt \
  --tag cov_cloth_q64_ownup_cont2_step23999_rg --roll_given > outputs/lsim/coverage_score_cont2_rg.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g2 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_${n}_step23999_train > outputs/lsim/coverage_score_${n}_train.log 2>&1 &
wait
echo "[given] $n scored $(date -u)"
