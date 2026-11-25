#!/bin/bash
# GPU 5: the deterministic baselines re-scored with area / strain (l2_cloth_long's new columns).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export CUDA_VISIBLE_DEVICES=5
$L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_q64X/step199.pt --data c1_long_test --frames 590 --tag long20_q64X_step199_shape > outputs/lsim/long20_q64X_step199_shape.log 2>&1 &
$L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_q64L/step1500.pt --data c1_long_test --frames 590 --tag long20_q64L_step1500_shape > outputs/lsim/long20_q64L_step1500_shape.log 2>&1 &
wait
$L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_q64X/step199.pt --tag long_q64X_step199_shape > outputs/lsim/long_q64X_step199_shape.log 2>&1 &
$L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_q64L/step1500.pt --tag long_q64L_step1500_shape > outputs/lsim/long_q64L_step1500_shape.log 2>&1 &
$L scripts/l2_cloth_long.py --ckpt still --data c1_long_test --frames 590 --tag long20_still_shape > outputs/lsim/long20_still_shape.log 2>&1 &
wait
echo "[done] $(date -u)"
