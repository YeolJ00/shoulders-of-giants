#!/bin/bash
# Final restoration config at inference: PGND for the real-fine-tuned 64M, the mixed model's 9 clips (GPU 6); the mixed
# model's shape over held-out episodes (GPU 7).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export LSIM_RESTORE=1
( CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt outputs/lsim/cloth_rq64/step1000.pt --tag rq64_step1000_P > outputs/lsim/pgnd_eval_rq64_step1000_P.log 2>&1
  bash outputs/lsim/render_mixed_P.sh outputs/lsim/cloth_m64/step23999.pt outputs/lsim/render_mixed_P0 6 ) &
( CUDA_VISIBLE_DEVICES=7 $L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64/step23999.pt --tag m64_P > outputs/lsim/shape_long_m64_P.log 2>&1
  CUDA_VISIBLE_DEVICES=7 $L scripts/l2_shape_long.py --ckpt outputs/lsim/cloth_m64R/step1999.pt --tag m64R_P > outputs/lsim/shape_long_m64R_P.log 2>&1 ) &
wait
echo "done $(date -u)"
