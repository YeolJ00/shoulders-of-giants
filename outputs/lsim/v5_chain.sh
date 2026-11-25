#!/bin/bash
# v5 (edge loss 100, from scratch, GPU 3): snapshots every 4,000 steps; when it ends, its full evaluation there
# (eval_cloth.sh: 96 rollouts, G5, G4, renders), the stretch and the real-cloth evaluation.   nohup outputs/lsim/v5_chain.sh &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until [ -f outputs/lsim/cloth_v5.log ]; do sleep 60; done
for s in 8000 12000 16000 20000; do
  until grep -q "^step *$((s + 50)) \|Traceback" outputs/lsim/cloth_v5.log; do sleep 60; done
  grep -q Traceback outputs/lsim/cloth_v5.log && { echo "[v5 chain] v5 failed $(date -u)"; exit 1; }
  cp outputs/lsim/cloth_v5/last.pt outputs/lsim/cloth_v5/step$s.pt; echo "[v5 chain] v5 step $s kept $(date -u)"
  CUDA_VISIBLE_DEVICES=3 $L scripts/l2_cloth_strain.py --ckpt outputs/lsim/cloth_v5/step$s.pt --tag v5_step$s > outputs/lsim/strain_v5_step$s.log 2>&1
done
until grep -q "^\[val\] step 23999\|Traceback" outputs/lsim/cloth_v5.log; do sleep 60; done
grep -q Traceback outputs/lsim/cloth_v5.log && { echo "[v5 chain] v5 failed $(date -u)"; exit 1; }
sleep 120                                             # the last save follows the last validation
echo "[v5 chain] v5 ended $(date -u)"
outputs/lsim/eval_cloth.sh outputs/lsim/cloth_v5/last.pt v5 data/datasets/c1_test 0 3
CUDA_VISIBLE_DEVICES=3 $L scripts/l2_cloth_strain.py --ckpt outputs/lsim/cloth_v5/last.pt --tag v5 > outputs/lsim/strain_v5.log 2>&1
CUDA_VISIBLE_DEVICES=3 $L scripts/l2_real_cloth_eval.py --ckpt outputs/lsim/cloth_v5/last.pt --tag v5 > outputs/lsim/real_eval_v5.log 2>&1
echo "[v5 chain] v5 evaluated $(date -u)"
