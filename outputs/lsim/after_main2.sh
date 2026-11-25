#!/bin/bash
# When training v0 ends: baselines B2 (GPU 1) and B1 (GPU 2), each optimised exactly like the
# model (16 windows x 2 accumulated = the model's 2 GPUs x 16, 15k steps, same noise and
# pushforward); the model's evaluation packed on GPU 1 next to B2: G1/G2/G4, Physics 101,
# drift, G3 sets.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
while ps aux | grep -q "[l]sim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 15000 --batch 16 --pair_norm --pf 3 --out outputs/lsim/train_v0"; do sleep 60; done
echo "[after_main] training v0 ended $(date -u)"; tail -3 outputs/lsim/train_v0.log
COMMON="--data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 15000 --batch 16 --accum 2 --pf 3"
CUDA_VISIBLE_DEVICES=1 setsid nohup $L -u -m lsim.train.train $COMMON --direct --out outputs/lsim/train_b2 > outputs/lsim/train_b2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 setsid nohup $L -u -m lsim.train.train $COMMON --arch gns --out outputs/lsim/train_b1 > outputs/lsim/train_b1.log 2>&1 &
echo "[after_main] B2 and B1 started $(date -u)"
C=outputs/lsim/train_v0/last.pt
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test --out docs/results/l2_eval_v0.json > outputs/lsim/eval_v0.log 2>&1
echo "[after_main] eval done $(date -u)"
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_phys101.py --ckpt $C --out docs/results/l2_phys101_v0.json > outputs/lsim/phys101_v0.log 2>&1
echo "[after_main] phys101 done $(date -u)"
CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_drift.py $C 24 free,tf,anch0.3 2>&1 | grep "^\[" > outputs/lsim/diag_final_v0.txt
echo "[after_main] drift done $(date -u)"
for s in mass3 poke2; do
    [ -d data/datasets/v0_test_$s ] && CUDA_VISIBLE_DEVICES=1 $L -u scripts/l2_eval.py --ckpt $C --data data/datasets/v0_test_$s \
        --out docs/results/l2_eval_v0_$s.json > outputs/lsim/eval_v0_$s.log 2>&1
done
echo "[after_main] all done $(date -u)"
