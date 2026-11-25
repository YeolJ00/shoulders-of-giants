#!/bin/bash
# The small-fit test, second round (2026-11-22). First round: the trainer on ONE episode fits that episode at 0.64 of
# constant velocity (cloth_fit_1: the full-data level) and its rollout of it explodes; in l2_overfit_batch 8 windows of
# that episode fit to 0.06 as they are and stall at 0.71 when re-collated every step with the trainer's augmentation
# (random yaw + shift). Here, on GPU 6 (empty twice a minute apart):
#   cloth_fit_1_noaug   the trainer on that episode with --no_aug --all_past (no noise, no pushforward), 4,000 steps,
#                       scored by l2_small_fit.py (its own episode one frame ahead; its rollout of it)
# and beside it, one after the other, l2_overfit_batch on the same 8 windows (1,200 steps, cosine): which part of the
# trainer's view costs the fit -- yaw alone, shift alone, a random shorter past, the trainer's full loss (side heads).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=6
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 120; done
echo "[noaug] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 0 --pf 0 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --save_every 1000 --keep --data data/datasets/c1 --shards 1"
( PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 1 \
    --no_aug --all_past --steps 4000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_noaug \
    > outputs/lsim/cloth_fit_1_noaug.log 2>&1
  echo "[noaug] 1 episode trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_noaug --episodes 1 > outputs/lsim/small_fit_1_noaug.log 2>&1
  echo "[noaug] 1 episode scored ($?) $(date -u)" ) &
for v in "yaw:--aug yaw" "shift:--aug shift" "hidepast:--hide_past" "full:--loss full --side_apart"; do
  t=${v%%:*}; o=${v#*:}
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit_batch.py --loss pos --episode 0 --n 8 --bs 4 --steps 1200 --sched cos $o \
    --tag ep0_w8_$t > outputs/lsim/overfit_ep0_w8_$t.log 2>&1
  echo "[noaug] 8 windows, $t: $(grep '^step  1199' outputs/lsim/overfit_ep0_w8_$t.log | cut -c1-70) $(date -u)"
done
wait
echo "[noaug] all done $(date -u)"
