#!/bin/bash
# Do the fully trained networks use the material they are given? (2026-11-20) l2_error_where --material on c1v_test
# (the windows ck1's 0.05-0.06 response was measured on): q64, the continuation and the stencil at step 23,999.
# Packed onto the GPU blind_chain.sh chose (it holds the teacher there; these are light).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until grep -q "\[blind\] GPU" outputs/lsim/blind_chain.log; do sleep 20; done
G=$(grep "\[blind\] GPU" outputs/lsim/blind_chain.log | awk '{print $3}')
echo "[mat] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
for r in q64 q64_cont q64_patch4; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step23999.pt --data c1v_test --n 300 --material \
    --tag ${r}_on_c1v_mat > outputs/lsim/where_${r}_on_c1v_mat.log 2>&1 &
done
wait
echo "[mat] done $(date -u)"
