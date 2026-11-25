#!/bin/bash
# When training v0 ends: B2 training (DDP, GPUs 1-2, same recipe with --direct) and, packed on
# GPU 1 next to it, the evaluation of the model: G1/G2/G4, G3 sets, Physics 101, drift.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
while ps aux | grep -q "[l]sim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 15000 --batch 16 --pair_norm --pf 3 --out outputs/lsim/train_v0"; do sleep 60; done
echo "[after_main] training v0 ended $(date -u)"; tail -3 outputs/lsim/train_v0.log
CUDA_VISIBLE_DEVICES=1,2 setsid nohup /home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun --nproc_per_node 2 --master_port 29532 \
    -m lsim.train.train --data data/datasets/v0,data/datasets/v0r --val data/datasets/v0_test --steps 15000 --batch 16 \
    --pf 3 --direct --out outputs/lsim/train_b2 > outputs/lsim/train_b2.log 2>&1 &
echo "[after_main] B2 started $(date -u)"
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
