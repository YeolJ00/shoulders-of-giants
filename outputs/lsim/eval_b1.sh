#!/bin/bash
# Baseline B1 (GNS, trained to 10k steps, ended 2026-10-20 00:09 UTC): the same evaluations as v1 and B2
# (G1/G2/G4, Physics 101, the drift diagnostic, the G3 sets), packed on GPU 3 next to the cloth gate.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
C=outputs/lsim/train_b1/last.pt
echo "[b1] evaluation start $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test --out docs/results/l2_eval_b1.json > outputs/lsim/eval_b1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_phys101.py --ckpt $C --out docs/results/l2_phys101_b1.json > outputs/lsim/phys101_b1.log 2>&1
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_drift.py $C 24 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_final_b1.txt
for s in mass3 poke2; do
    CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test_$s --out docs/results/l2_eval_b1_$s.json > outputs/lsim/eval_b1_$s.log 2>&1
done
echo "[b1] evaluation done $(date -u)"
