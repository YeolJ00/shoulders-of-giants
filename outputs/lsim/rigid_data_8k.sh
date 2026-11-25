#!/bin/bash
# Does more rigid data help the network run alone? (2026-11-25). The plain rigid transformer, every input given, at
# 36,000 steps (OneCycle over the run, so a checkpoint of a longer run is not this): 128 -> 1,024 episodes left one
# frame level (0.454 -> 0.447 on shard 8, -0.012 [-0.045, +0.019]), closed the train / unseen gap (own 0.411 -> 0.476)
# and made replays closer (bodies that moved: 5.8 -> 2.9 mm median at 2 s, closer in 76%). Three times the steps on
# the same 1,024 (108,000) took one frame to 0.368 but replays of moved bodies ended farther off (+8.3 [+4.1, +13.8]
# mm at the end): one-frame fit and running alone part with longer training, so what makes the rigid network run
# alone well is the open fit question, and more episodes is the lever that has helped it. Two arms at 36,000 steps,
# each against rigid_pilot1024p_attn_direct_36k (same recipe, 1,024 episodes, no yaw):
#   rigid_pilot8192p_attn_direct_36k       --episodes 8192   (shards 0-63)                  GPU B
#   rigid_pilot1024p_attn_direct_yaw_36k   --episodes 1024 --aug 1  (the random yaw and shift; at 128 episodes it
#                                          acted as extra data: own 0.411 -> 0.462, unseen -0.013 / -0.029, replays
#                                          closer at the end by 6.9 mm)                       GPU A
# Shard 8 is inside the 8,192 arm's training set, so all are scored on shard 64 (the first shard none trained on),
# 32 episodes, with the existing 1,024 x 36k and 1,024 x 108k networks scored there first; the 1,024 arms also on
# shard 8; every arm on its own first 32 episodes (shard 0).
# PREDICTION (written before the run):
#   8,192: one frame on shard 64 within 0.02 of the 1,024 run's (steps set it); own within 0.02 of unseen (no gap);
#          replays of bodies that moved closer -- median at 2 s 15-30% lower, end mean lower by 3-10 mm, closer in
#          60%+ of episodes.
#   yaw:   one frame level (within 0.02), own a little worse than without (less memorised, 0.48-0.52), replays at
#          the end closer by 0-5 mm (at 1,024 episodes the yaw has less to add than at 128).
# READ: 8,192 closer on replays = episodes are the lever for running alone and the rigid scale-up uses all 50,048
# (with long training next); level = 1,024 episodes are enough for this network at this budget and the replay gap to
# the teacher is not a data question. The yaw closer = it joins the scale-up (it costs nothing on one frame).
#   usage: rigid_data_8k.sh <gpu A, yaw arm> <gpu B, 8,192 arm>   (each empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
wait_gpu() {
  local u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$1 '$1==g{print $2}')
  busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
  while busy || { sleep 60; busy; }; do sleep 60; done
}
score() {   # name gpu tag shard
  CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$1.pt --episodes 32 --shard $4 --tag $3 \
    > outputs/lsim/small_fit_${1}_$3.log 2>&1
}
train() {   # name gpu extra-args...
  local n=$1 g=$2; shift 2
  echo "[data8k] $n on GPU $g $(date -u)"
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --steps 36000 --pair_norm --coulomb --pf 6 \
    --noise 0.003,0.015 --mix --plain --trunk attn --d 128 --layers 6 --direct --out outputs/lsim/$n.pt "$@" \
    > outputs/lsim/$n.log 2>&1
  echo "[data8k] $n trained ($?) $(date -u)"
}
A=$1; B=$2
wait_gpu $A; wait_gpu $B
echo "[data8k] GPUs $A $B $(date -u)"
# the references on the new unseen shard
score rigid_pilot1024p_attn_direct_36k $A held64 64 & score rigid_pilot1024p_attn_direct_108k $B held64 64 &
# smoke: 8,192 episodes load and train (RAM, step time)
CUDA_VISIBLE_DEVICES=$B $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 8192 --steps 30 --pair_norm --coulomb \
  --pf 6 --noise 0.003,0.015 --mix --plain --trunk attn --d 128 --layers 6 --direct \
  --out outputs/lsim/tmp/smoke_data8k.pt > outputs/lsim/tmp/smoke_data8k.log 2>&1 \
  && echo "[data8k] smoke ok $(date -u)" || { echo "[data8k] smoke FAILED $(date -u)"; wait; exit 1; }
rm -f outputs/lsim/tmp/smoke_data8k.pt
wait
echo "[data8k] references scored $(date -u)"
{ train rigid_pilot1024p_attn_direct_yaw_36k $A --episodes 1024 --aug 1
  n=rigid_pilot1024p_attn_direct_yaw_36k
  score $n $A own 0 & score $n $A held 8 & score $n $A held64 64 & wait
  echo "[data8k] $n scored $(date -u)"; } &
{ train rigid_pilot8192p_attn_direct_36k $B --episodes 8192
  n=rigid_pilot8192p_attn_direct_36k
  score $n $B own 0 & score $n $B held64 64 & wait
  echo "[data8k] $n scored $(date -u)"; } &
wait
echo "[data8k] done $(date -u)"
