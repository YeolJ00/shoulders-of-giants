#!/bin/bash
# v1 tests (GPUs 3-4): the Coulomb default against the support default (recipe T), each trained on
# 100 episodes of v0 (mostly resting and pushed bodies) or of v0r (ramps, where things must slide).
# After each run: the 30-frame drift diagnostic on its own training data and on v0_test, then the
# 93 real Physics 101 slides. ofT (recipe T on v0, trained 2026-10-17) is re-scored the same way.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
R="--episodes 100 --steps 2000 --batch 4 --pair_norm --pf 6 --noise 0.003,0.015 --mix"
score() {  # gpu name data
    CUDA_VISIBLE_DEVICES=$1 $L -u scripts/l2_drift.py outputs/lsim/$2.pt 12 free,tf $3 2>&1 | grep "^\[" > outputs/lsim/drift_$2.txt
    CUDA_VISIBLE_DEVICES=$1 $L -u scripts/l2_phys101.py --ckpt outputs/lsim/$2.pt --out outputs/lsim/phys101_$2.json > outputs/lsim/phys101_$2.log 2>&1
    echo "[v1 tests] $2 scored $(date -u)"
}
run() {    # gpu name data prior-flag
    CUDA_VISIBLE_DEVICES=$1 $L -u scripts/l2_overfit.py --data $3 $R $4 --out outputs/lsim/$2.pt > outputs/lsim/$2.log 2>&1
    echo "[v1 tests] $2 trained $(date -u)"
    score $1 $2 $3
}
echo "[v1 tests] start $(date -u)"
run 3 ofV1  data/datasets/v0  --coulomb &
run 3 ofTr  data/datasets/v0r --support &
run 4 ofV1r data/datasets/v0r --coulomb &
score 4 ofT data/datasets/v0 &
wait
echo "[v1 tests] all done $(date -u)"
