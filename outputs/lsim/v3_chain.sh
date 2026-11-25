#!/bin/bash
# After v3 (layer 3) ends on GPU 3: its full evaluation there (eval_cloth.sh: 96 rollouts, G5, G4, renders), and
# snapshots every 4,000 steps on the way.   nohup outputs/lsim/v3_chain.sh &
cd /home/nas5/jooyeolyun/repos/simulation-assestization
until [ -f outputs/lsim/cloth_v3.log ]; do sleep 60; done
for s in 8000 12000 16000 20000; do
  until grep -q "^step *$((s + 50)) \|Traceback" outputs/lsim/cloth_v3.log; do sleep 60; done
  grep -q Traceback outputs/lsim/cloth_v3.log && { echo "[v3 chain] v3 failed $(date -u)"; exit 1; }
  cp outputs/lsim/cloth_v3/last.pt outputs/lsim/cloth_v3/step$s.pt; echo "[v3 chain] v3 step $s kept $(date -u)"
done
until grep -q "^\[val\] step 23999\|Traceback" outputs/lsim/cloth_v3.log; do sleep 60; done
grep -q Traceback outputs/lsim/cloth_v3.log && { echo "[v3 chain] v3 failed $(date -u)"; exit 1; }
sleep 120                                             # the last save follows the last validation
echo "[v3 chain] v3 ended $(date -u)"
outputs/lsim/eval_cloth.sh outputs/lsim/cloth_v3/last.pt v3 data/datasets/c1_test 0 3
echo "[v3 chain] v3 evaluated $(date -u)"
