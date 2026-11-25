#!/bin/bash
# Training v1 (DDP, GPUs 3-4): recipe T with the Coulomb default in place of the support default.
# Alone on both cards while it trains (rank 0 of v0 held 42 GB). Then: B1 resumes from its step-2000
# checkpoint on GPU 4 (same flags as at its start), and v1 and B2 are evaluated side by side on GPU 3
# (the same evaluations as model v0: G1/G2/G4, Physics 101, the drift diagnostic, the G3 sets).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
DATA="--data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 10000 --batch 16"
RECIPE="--pf 6 --noise 0.003,0.015 --mix"
echo "[v1] training $(date -u): $DATA $RECIPE --pair_norm --coulomb"
CUDA_VISIBLE_DEVICES=3,4 /home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun --nproc_per_node 2 --master_port 29541 \
    -m lsim.train.train $DATA $RECIPE --pair_norm --coulomb --out outputs/lsim/train_v1 > outputs/lsim/train_v1.log 2>&1
echo "[v1] training ended $(date -u)"; tail -2 outputs/lsim/train_v1.log
CUDA_VISIBLE_DEVICES=4 setsid nohup $L -u -m lsim.train.train $DATA --accum 2 $RECIPE --arch gns --resume \
    --out outputs/lsim/train_b1 >> outputs/lsim/train_b1.log 2>&1 < /dev/null &
echo "[v1] B1 resumed on GPU 4 $(date -u)"
evaluate() {   # name checkpoint
    CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_eval.py --ckpt $2 --data data/datasets/v0_test --out docs/results/l2_eval_$1.json > outputs/lsim/eval_$1.log 2>&1
    CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_phys101.py --ckpt $2 --out docs/results/l2_phys101_$1.json > outputs/lsim/phys101_$1.log 2>&1
    CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_drift.py $2 24 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_final_$1.txt
    for s in mass3 poke2; do
        CUDA_VISIBLE_DEVICES=3 $L -u scripts/l2_eval.py --ckpt $2 --data data/datasets/v0_test_$s --out docs/results/l2_eval_$1_$s.json > outputs/lsim/eval_$1_$s.log 2>&1
    done
    echo "[v1] $1 evaluated $(date -u)"
}
evaluate v1 outputs/lsim/train_v1/last.pt &
evaluate b2 outputs/lsim/train_b2/last.pt &
wait
echo "[v1] evaluation done $(date -u)"
