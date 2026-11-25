#!/bin/bash
# The pushforward trial's cloth result, looked at (2026-11-25). Practising longer alone in training made the blow-up
# tail grow with the horizon: 25 / 31 / 44 of 336 held-out rollouts above 2x area at pf 4 / 8 / 16 (step 2,999), one
# frame level (+0.005 / +0.006). Render three held-out episodes where pf 16 ends at 14-17x area and the control keeps
# the cloth (0.94-0.98x), both networks, material given as they were scored: fold shard 4 ep 8, lift-drop shard 11
# ep 16, shake shard 3 ep 28.
# PREDICTION: the pf 16 cloth stretches from the moment it is released or lands (a fast local tear-like stretch that
# then spreads), not a slow uniform swelling from the first frame. Evaluation only, one GPU.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_pf
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[pfr] GPU $g $(date -u)"
mkdir -p $O
for ep in "shard_00004 8" "shard_00011 16" "shard_00003 28"; do
  set -- $ep
  for x in "cloth_pf_ctl pfctl" "cloth_pf16 pf16"; do
    set -- $ep $x
    D=$O/${1}_e${2}_t7_$4
    ( CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$3/step2999.pt data/datasets/c1_test/$1 $2 7 end $4 --out $O
      CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
      $L scripts/l2_render_cloth.py compose $D --out $O
      echo "[pfr] render $1 $2 $4 ($?) $(date -u)" ) &
  done
done
wait
echo "[pfr] done $(date -u)"
