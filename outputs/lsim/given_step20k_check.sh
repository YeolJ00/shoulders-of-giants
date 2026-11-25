#!/bin/bash
# A second checkpoint for the every-input-given result (2026-11-25; the rule: never judge on one checkpoint). At step
# 23,999 the third stretch with every input given (cloth_q64_ownup_cont2_given) keeps the cloth where the hidden-input
# stretch (cloth_q64_ownup_cont2) does not: blow-ups 5 against 28 of 336, strain mean 4.2% against 24.3%, one frame
# 0.475 against 0.486, positions level -- both rolled out with the material given. Score both at step 20,000 the
# same way (cont2's step 20,000 was scored with the material hidden in cont_tail_check.sh: 27 blow-ups).
# PREDICTION: at 20,000 the given network has at most 10 blow-ups and the old one at least 20; one frame within 0.01
# of each other (the given one ahead or level). If the given network's tail at 20,000 is as large as the old one's,
# the 23,999 result is a checkpoint swing, not the recipe.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[g20] GPU $g $(date -u)"
for n in cloth_q64_ownup_cont2_given cloth_q64_ownup_cont2; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step20000.pt --tag cov_${n}_step20000_rg \
    --roll_given > outputs/lsim/coverage_score_${n}_step20000_rg.log 2>&1 &
done
wait
echo "[g20] done $(date -u)"
