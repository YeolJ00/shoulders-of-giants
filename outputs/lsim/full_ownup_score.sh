#!/bin/bash
# Score one checkpoint of the full-data --own_up run (cloth_q64_ownup) on c1_test, one frame only, and print the paired
# comparison against cloth_q64 at the same step (docs/results/l2_coverage_cov_full_q64_step<N>.json; same 1,400 windows).
#   $1 = step (8000, 16000); waits for the checkpoint, then for a GPU with no compute app twice a minute apart.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
S=/tmp/claude-1000021/-home-nas5-jooyeolyun-repos-simulation-assestization/b6ae3a69-1471-487b-b0b6-6b5e1ac4d459/scratchpad
n=$1
until [ -f outputs/lsim/cloth_q64_ownup/step$n.pt ] || grep -a -q "Traceback" outputs/lsim/cloth_q64_ownup.log; do sleep 120; done
[ -f outputs/lsim/cloth_q64_ownup/step$n.pt ] || { echo "the run failed"; grep -a -A12 "Traceback" outputs/lsim/cloth_q64_ownup.log | tail -30; exit 1; }
sleep 30
free() { u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1==g{print $2}'); ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
g=""
while [ -z "$g" ]; do
  for c in 2 3 0 1 4 5; do if free $c && sleep 60 && free $c; then g=$c; break; fi; done
  [ -z "$g" ] && sleep 120
done
echo "gpu $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_q64_ownup/step$n.pt --n_roll 0 \
  --tag cov_full_q64_ownup_step$n > outputs/lsim/coverage_score_q64_ownup_$n.log 2>&1
$L $S/paired_cov.py cov_full_q64_step$n cov_full_q64_ownup_step$n
grep -a '\[val\]' outputs/lsim/cloth_q64_ownup.log | tail -8 | cut -c1-120
date -u
