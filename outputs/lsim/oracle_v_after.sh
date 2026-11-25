#!/bin/bash
# After the oracle-velocity chain: rescore the positions-only control at step 7,999 with per-window sums (paired
# interval), then the figure.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until grep -q "scored\|failed" outputs/lsim/oracle_v_chain.log || ! ps aux | grep -q "[o]racle_v_chain"; do sleep 60; done
grep -q "scored" outputs/lsim/oracle_v_chain.log || { echo "[after] chain did not finish"; exit 1; }
CUDA_VISIBLE_DEVICES=7 $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1v_test --n 300 --tag ck1_on_c1v \
  > outputs/lsim/where_ck1_on_c1v.log 2>&1
CUDA_VISIBLE_DEVICES="" $L scripts/l2_oracle_v_plot.py --out /tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad/oracle_v.png
echo "[after] done"
