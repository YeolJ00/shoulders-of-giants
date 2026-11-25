#!/bin/bash
# Shape over whole held-out episodes (scripts/l2_shape_long.py) for the mixed model: as trained, with the restoration at
# inference, and the edge-loss + rollout fine-tune (m64R); GPU 7.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export CUDA_VISIBLE_DEVICES=7
$L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64/step23999.pt --tag m64 > outputs/lsim/shape_long_m64.log 2>&1
LSIM_RESTORE=1 $L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64/step23999.pt --tag m64_P > outputs/lsim/shape_long_m64_P.log 2>&1
$L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64R/step1999.pt --tag m64R > outputs/lsim/shape_long_m64R.log 2>&1
