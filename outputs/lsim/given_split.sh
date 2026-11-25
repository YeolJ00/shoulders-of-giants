#!/bin/bash
# Which given input blows up the cloth rollouts (2026-11-24). cloth_cov_seven_ownup_given (--given: no slot hidden, the
# grasp force never hidden, all 8 past frames) against its twin cloth_cov_seven_ownup (slots hidden with p 0.5, the
# grasp force hidden in 30% of windows, a random shorter past): one frame 0.764 against 0.753 (+0.012 [+0.002, +0.021];
# predicted within 0.02: held), but rollouts on the same 336 held-out episodes 64.3 / 127.0 / 153.5 mm against
# 25.6 / 78.8 / 115.4 (+38.7 [+33.7, +44.2] at 1 s, worse in 93%), strain 215% against 61%, area 5.9x against 2.2x:
# the cloth blows up (predicted level: failed). Two arms, each the twin's recipe with one part of --given:
#   cloth_cov_seven_ownup_given_rp   --given --random_past: slots and grasp force given, the random shorter past kept
#   cloth_cov_seven_ownup_allpast    --all_past: slots and grasp force hidden as in the twin, all 8 past frames always
# PREDICTION (written before the run): the shorter past is what keeps rollouts stable (a network that always has 8
# frames leans on its older frames, which in a rollout are its own drifting output; pushforward practises only 4):
#   given_rp: rollouts level with the twin (1 s 24-30 mm, strain <= 80%); one frame within 0.02 of 0.753;
#   allpast:  rollouts like --given's (1 s 50-70 mm, strain >= 150%); one frame within 0.02 of 0.753.
# READ: as predicted = the cloth recipe keeps the random shorter past (an input that is not hidden: it trains the
# network not to trust its history), slots and grasp force given; the reverse = the hidden slots or grasp force are
# the stabiliser; both level with the twin = the two interact.
#   $1 = arm (given_rp | allpast), $2 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
arm=$1; g=$2; n=cloth_cov_seven_ownup_$arm
case $arm in given_rp) X="--given --random_past";; allpast) X="--all_past";; *) echo "unknown arm $arm"; exit 1;; esac
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[split] $n ($X) GPU $g $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --side_apart --own_up $X --keep --data data/datasets/c1 --sig 1e-3 --pf 4 --steps 12000 --save_every 3000 \
  --subset outputs/lsim/coverage_subsets/seven.json --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[split] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step11999.pt \
  --tag cov_seven_ownup_${arm}_step11999 > outputs/lsim/coverage_score_seven_ownup_$arm.log 2>&1
echo "[split] $n scored $(date -u)"
