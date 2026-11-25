#!/bin/bash
# c1 regenerated after the teacher fixes (clearance, guarded placing, edge-edge self-contact, damped contact, fp32):
# half the shards on each of GPUs 3 and 4 (finished shards are skipped), then the held-out split on GPU 3
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
echo "[c1] start $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 20000 --worlds 64 --seed 0 --part 0/2 > outputs/lsim/gen_c1v2_p0.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1 --episodes 20000 --worlds 64 --seed 0 --part 1/2 > outputs/lsim/gen_c1v2_p1.log 2>&1 &
wait
echo "[c1] train done $(date -u)"
CUDA_VISIBLE_DEVICES=3 $L -u scripts/l1_generate_cloth.py --out data/datasets/c1_test --episodes 1024 --worlds 64 --seed 1 --split test --part 0/1 > outputs/lsim/gen_c1v2_test.log 2>&1
echo "[c1] test done $(date -u)"
