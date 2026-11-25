#!/bin/bash
# After the full local-frame run's final score (2026-11-23: cloth_q64_ownup step 23,999 on c1_test 0.588 against the
# original's 0.688 and 0.667 after twice the training; rollouts 17.6 / 64.7 / 97.0 mm against 24.4 / 79.6 / 121.7):
#   1. its one-frame score on episodes it trained on (the seven arm's 2,000, as for the other runs): the train /
#      held-out gap that "scale beyond 20k?" reads (tag cov_cloth_q64_ownup_step23999_train)
#   2. the four held-out episodes of render_heldout_pair.sh again, with the two full-data networks: the original
#      trained twice as long (cloth_q64_cont, tag origfull) and the local-frame one (tag localfull), each alone from
#      frame 7 to the end beside the teacher. Clips in outputs/lsim/render_heldout_full/.
#   $1 = GPU (waits until it has no compute app twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_full
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
mkdir -p $O
echo "[extras] GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth_q64_ownup/step23999.pt --n_roll 0 --data c1 \
  --subset outputs/lsim/coverage_subsets/seven.json --tag cov_cloth_q64_ownup_step23999_train > outputs/lsim/coverage_score_cloth_q64_ownup_train.log 2>&1 &
for ep in "shard_00008 21" "shard_00008 17" "shard_00000 18" "shard_00008 50"; do
  set -- $ep
  for net in "cloth_q64_cont/step23999.pt origfull" "cloth_q64_ownup/step23999.pt localfull"; do
    n=${net% *}; t=${net#* }
    D=$O/${1}_e${2}_t7_$t
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$n data/datasets/c1_test/$1 $2 7 end $t --out $O
    CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
    $L scripts/l2_render_cloth.py compose $D --out $O
    echo "[extras] $1 $2 $t ($?) $(date -u)"
  done
done
wait
echo "[extras] done $(date -u)"
