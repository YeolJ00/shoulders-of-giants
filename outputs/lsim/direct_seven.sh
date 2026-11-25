#!/bin/bash
# Does the cloth network fit better WITHOUT its exact layer? (2026-11-23) The rigid track's finding of the day: the
# rigid layer alone is 1.74 of constant velocity over one frame, every network trained on it inherited that, and a
# plain transformer with no layer (zero output = constant velocity) on 128 episodes matched the baselines trained on
# 50,048. The same measurement on cloth (l2_coverage_score on cloth_q64_ownup with its force heads zeroed, tag
# cov_cloth_layer_alone): THE CLOTH LAYER ALONE IS AT 2.22 of constant velocity (by family 1.93-2.85) -- free fall and
# surface contact, no cloth forces -- and the best network gets from there to 0.588. The graph-era direct baseline
# (B2, October) was ahead on one frame (0.69 against 0.75 mm) and behind at rest and over a second (4.2 against 0.05
# mm; 39 against 29 mm); the current network has never been trained --direct. Here: cloth_cov_seven_ownup's exact
# twin (2,000 episodes of all seven families, 12,000 steps x 1 GPU x batch 4, --own_up, the noise and pushforward)
# + --direct (cloth_qwen: the whole acceleration in g, held vertices given, nothing else exact):
#   cloth_cov_seven_ownup_direct     against cloth_cov_seven_ownup (one frame 0.949 / 0.866 / 0.780 / 0.753 at 3,000 /
#   6,000 / 9,000 / 11,999; rollouts on 336 held-out episodes 25.6 / 78.8 / 115.4 mm, strain median 11%)
# PREDICTION (written before the run): one frame ahead at every checkpoint (it starts at 1.0, the twin at 2.2) and
# ends 0.66-0.73 (-0.02 to -0.09 paired); rollouts WORSE at 1 s (30-45 mm) because nothing holds resting cloth or
# keeps it out of the table, and strain no better. READ: ahead on one frame by >= 0.03 at the end = the layer costs
# cloth fit too, and what it buys (rest, contact, real cloth's shape) has to be weighed as for rigid; level or behind
# = for cloth the layer is not what limits the fit (the network learns the internal forces either way).
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=cloth_cov_seven_ownup_direct
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[direct7] GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up --direct --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --subset outputs/lsim/coverage_subsets/seven.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[direct7] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step11999.pt \
  --tag cov_seven_ownup_direct_step11999 > outputs/lsim/coverage_score_seven_ownup_direct.log 2>&1 &
for s in 3000 6000 9000; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step$s.pt --n_roll 0 \
    --tag cov_seven_ownup_direct_step$s > outputs/lsim/coverage_score_seven_ownup_direct_$s.log 2>&1 &
done
wait
echo "[direct7] scored $(date -u)"
