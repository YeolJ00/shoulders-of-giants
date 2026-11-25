#!/bin/bash
# The fully trained stencil network and the continuation against the local rule on its own vertices (2026-11-20):
# 2,000 c1_test windows (probe R 6), as q64's 0.667 against the MLP's 0.68. Packed onto blind_chain.sh's GPU.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until grep -q "\[blind\] GPU" outputs/lsim/blind_chain.log; do sleep 20; done
G=$(grep "\[blind\] GPU" outputs/lsim/blind_chain.log | awk '{print $3}')
echo "[p2k] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
for r in q64_cont q64_patch4; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step23999.pt --data c1_test --n 2000 --probe_R 6 \
    --tag ${r}_on_c1_2000R6 > outputs/lsim/where_${r}_on_c1_2000R6.log 2>&1 &
done
wait
echo "[p2k] done $(date -u)"
