#!/bin/bash
# G3 test sets (masses x3, unseen pokes x2) on the same held-out scenes as v0_test (seed 1),
# on GPU 2 once the chain's dataset checks are done (packed next to B2 training).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until grep -q "checks done" outputs/lsim/after_data.log 2>/dev/null; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_generate.py --out data/datasets/v0_test_mass3 --episodes 256 --worlds 128 --seed 1 \
    --split test --mass_scale 3 > outputs/lsim/gen_v0_test_mass3.log 2>&1
CUDA_VISIBLE_DEVICES=2 $L -u scripts/l1_generate.py --out data/datasets/v0_test_poke2 --episodes 256 --worlds 128 --seed 1 \
    --split test --poke_scale 2 > outputs/lsim/gen_v0_test_poke2.log 2>&1
echo "[g3] done $(date)"
