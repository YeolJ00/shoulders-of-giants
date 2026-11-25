#!/bin/bash
# Training v0 (DDP, GPUs 1-2) with the recipe flags given as arguments, then baselines B2 (GPU 1)
# and B1 (GPU 2) with the same recipe (minus the model-only flags), and the model's evaluation
# packed next to B2 on GPU 1.   usage: launch_v0.sh "<recipe flags>" "<model-only flags>"
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
RECIPE="$1"; MODEL="$2"
DATA="--data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 10000 --batch 16"
echo "[launch] model $(date -u): $DATA $RECIPE $MODEL"
CUDA_VISIBLE_DEVICES=1,2 /home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun --nproc_per_node 2 --master_port 29533 \
    -m lsim.train.train $DATA $RECIPE $MODEL --out outputs/lsim/train_v0 > outputs/lsim/train_v0.log 2>&1
echo "[launch] model training ended $(date -u)"; tail -2 outputs/lsim/train_v0.log
CUDA_VISIBLE_DEVICES=1 setsid nohup $L -u -m lsim.train.train $DATA --accum 2 $RECIPE --direct --out outputs/lsim/train_b2 > outputs/lsim/train_b2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup $L -u -m lsim.train.train $DATA --accum 2 $RECIPE --arch gns --out outputs/lsim/train_b1 > outputs/lsim/train_b1.log 2>&1 &
echo "[launch] B2 and B1 started $(date -u)"
C=outputs/lsim/train_v0/last.pt
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test --out docs/results/l2_eval_v0.json > outputs/lsim/eval_v0.log 2>&1
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_phys101.py --ckpt $C --out docs/results/l2_phys101_v0.json > outputs/lsim/phys101_v0.log 2>&1
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_drift.py $C 24 free,tf 2>&1 | grep "^\[" > outputs/lsim/diag_final_v0.txt
for s in mass3 poke2; do
    CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test_$s --out docs/results/l2_eval_v0_$s.json > outputs/lsim/eval_v0_$s.log 2>&1
    CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_eval.py --ckpt cv --data data/datasets/v0_test_$s --out docs/results/l2_eval_b0_$s.json > outputs/lsim/eval_b0_$s.log 2>&1
done
echo "[launch] model evaluation done $(date -u)"
