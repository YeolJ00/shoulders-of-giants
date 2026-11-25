#!/bin/bash
# Score the stencil run and the original at step 2000 (and the original at 4000 with the probe subset) on c1v_test, on
# the GPU the probe-scale chain already holds (light jobs packed onto one card). Arg: the GPU index.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export CUDA_VISIBLE_DEVICES=$1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_patch4/step2000.pt --data c1v_test --n 300 --tag patch4_2000 > outputs/lsim/where_patch4_2000.log 2>&1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step2000.pt --data c1v_test --n 300 --tag ck1_on_c1v_2000 > outputs/lsim/where_ck1_2000.log 2>&1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step4000.pt --data c1v_test --n 300 --tag ck1_on_c1v_4000p > outputs/lsim/where_ck1_4000p.log 2>&1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1v_test --n 300 --tag ck1_on_c1v_p > outputs/lsim/where_ck1_p.log 2>&1
echo "[early] done $(date -u)"
