#!/bin/bash
# Watch the 2,000-episode pair on held-out episodes (2026-11-22): the world-frame network (cloth_cov_seven) and its
# --own_up twin (cloth_cov_seven_ownup), both at step 11,999, each rolling out the same c1_test episodes alone from
# frame 7 to the end, beside the teacher. Episodes: the first rollout row of fold, shake, lift_drop and push in
# l2_coverage_score's seeded list (docs/results/l2_coverage_cov_seven_step11999.json) -- fixed before the --own_up
# network's rollouts of them were read. Clips in outputs/lsim/render_heldout_pair/, tags orig2k / local2k.
#   $1 = GPU (waits until it has no compute app twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
O=outputs/lsim/render_heldout_pair
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
mkdir -p $O
echo "[pairvid] GPU $g $(date -u)"
for ep in "shard_00008 21" "shard_00008 17" "shard_00000 18" "shard_00008 50"; do
  set -- $ep
  for net in "cloth_cov_seven orig2k" "cloth_cov_seven_ownup local2k"; do
    n=${net% *}; t=${net#* }
    D=$O/${1}_e${2}_t7_$t
    CUDA_VISIBLE_DEVICES=$g $L scripts/l2_render_cloth.py export outputs/lsim/$n/step11999.pt data/datasets/c1_test/$1 $2 7 end $t --out $O
    CUDA_VISIBLE_DEVICES=$g $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
    $L scripts/l2_render_cloth.py compose $D --out $O
    echo "[pairvid] $1 $2 $t ($?) $(date -u)"
  done
done
echo "[pairvid] done $(date -u)"
