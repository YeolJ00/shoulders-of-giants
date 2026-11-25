#!/bin/bash
# Overnight 2026-10-20/21 (second version): snapshot v2 every 4,000 steps; when B2 ends, v3 (layer 3) trains on GPU 3;
# when v2 ends, on GPU 4: B2's full evaluation, v2's, then v2's 8k / 12k snapshots on the same 96 starts as v1's.
#   nohup outputs/lsim/night_chain.sh <v2 pid> <b2 pid> &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
V2=$1; B2=$2
(for s in 8000 12000 16000 20000; do                 # last.pt is saved after step s's validation, before s + 50 logs
   until grep -q "^step *$((s + 50)) " outputs/lsim/cloth_v2.log; do sleep 60; done
   cp outputs/lsim/cloth_v2/last.pt outputs/lsim/cloth_v2/step$s.pt; echo "[chain] v2 step $s kept $(date -u)"
 done) &
(while kill -0 $B2 2>/dev/null; do sleep 60; done
 echo "[chain] B2 ended $(date -u); v3 starts on GPU 3"
 outputs/lsim/train_cloth_v3.sh 24000 3
 echo "[chain] v3 ended $(date -u)") &
(while kill -0 $V2 2>/dev/null; do sleep 60; done
 echo "[chain] v2 ended $(date -u)"
 until ! kill -0 $B2 2>/dev/null; do sleep 60; done
 outputs/lsim/eval_cloth.sh outputs/lsim/cloth_b2/last.pt b2 data/datasets/c1_test 0 4
 echo "[chain] B2 evaluated $(date -u)"
 outputs/lsim/eval_cloth.sh outputs/lsim/cloth_v2/last.pt v2 data/datasets/c1_test 0 4
 echo "[chain] v2 evaluated $(date -u)"
 for s in 8000 12000; do
   CUDA_VISIBLE_DEVICES=4 $L scripts/l2_cloth_eval.py --ckpt outputs/lsim/cloth_v2/step$s.pt --data data/datasets/c1_test \
     --shards 16 --n 96 --tag v2_step$s > outputs/lsim/eval_v2_step$s.log 2>&1
   echo "[chain] v2 step $s evaluated $(date -u)"
 done) &
wait
echo "[chain] all done $(date -u)"
