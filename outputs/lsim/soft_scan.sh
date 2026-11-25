#!/bin/bash
# Soft-body restoration variants at inference on the mixed model (8 held-out s1_test episodes each; GPU 7).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export CUDA_VISIBLE_DEVICES=7
for spec in 1 iters_jacobi=80 soft=0.2:0.2 vol=0.05:0.2 soft=0.2:0.2,vol=0.05:0.2,iters_jacobi=80 pre=1; do
  LSIM_RESTORE=$spec $L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64/step23999.pt --data s1_test --n 8 --tag scratch_soft > outputs/lsim/soft_scan_$spec.log 2>&1
  echo "$spec $(grep '^==' outputs/lsim/soft_scan_$spec.log)"
done
