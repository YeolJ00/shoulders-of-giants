#!/bin/bash
# The small-fit test, the replay, --own_up at 4,000 steps (2026-11-22). One episode with chunk noise + pushforward:
#   world frame, no augmentation (cloth_fit_1_noaug_pf)   frames 0.160   replay  8.7 /  4.8 / 17.2 mm at 1 s / 2 s / end
#   --own, the trainer as it is  (cloth_fit_1_own_pf)     frames 0.103   replay 12.0 / 39.9 / 93.3 mm, crumpled at the end
# The --own network fits the frames better and replays worse. Suspect: what it lacks on held-out cloth -- the
# vertical (it did not cancel gravity at tilts it had not seen; a replay's drift is such tilts). Here the same run
# with --own_up:
#   cloth_fit_1_ownup_pf   1 episode, 4,000 steps, the trainer as it is, --sig 1e-3 --pf 4
# GPU $1 once it is empty twice a minute apart; scored by l2_small_fit.py.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[ownup_pf1] GPU $g $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --own_up --save_every 1000 --keep --data data/datasets/c1 --shards 1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $NET --episodes 1 \
  --steps 4000 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_1_ownup_pf \
  > outputs/lsim/cloth_fit_1_ownup_pf.log 2>&1
echo "[ownup_pf1] trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit.py --run cloth_fit_1_ownup_pf --episodes 1 > outputs/lsim/small_fit_1_ownup_pf.log 2>&1
echo "[ownup_pf1] scored ($?) $(date -u)"
