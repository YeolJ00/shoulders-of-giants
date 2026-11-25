#!/bin/bash
# Can the network fit a handful of teacher episodes at all (2026-11-22; the user: "can we even fit the teacher on small
# data?")? The 64M network, q64's recipe + --side_apart, from scratch, NO noise and NO pushforward (--sig 0 --pf 0:
# plain memorisation), on the first 1 and the first 16 episodes of c1's first shard:
#   cloth_fit_1    1 episode   4,000 steps x 4 windows (each of its ~105 windows ~150 times)
#   cloth_fit_16   16 episodes 8,000 steps x 4 windows (each of ~1,700 windows ~19 times)
# Scored by scripts/l2_small_fit.py: one frame on its own episodes and on held-out ones at every checkpoint, the last
# checkpoint rolled out on the episodes it trained on. Earlier evidence (l2_overfit_batch, 2026-11-10): 16 FIXED windows
# shown 1,250 times each still at 0.50 mm (0.065-1.15 by batch; the untrained network 5.9), two architectures alike.
# Takes the first GPU(s) found empty three times over 2.5 minutes: two -> both runs at once; one -> one after the other.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' ')
  sleep 75
  b=$(empty_gpus | tr '\n' ' ')
  sleep 75
  c=$(empty_gpus | tr '\n' ' ')                                   # empty three times over 2.5 min: a coverage arm's
  ab=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort))   # scorer takes ~1 min to reach its GPU
  G=$(comm -12 <(echo "$ab") <(echo $c | tr ' ' '\n' | sort) | sort -n | head -2 | tr '\n' ' ')
  [ -z "$(echo $G | tr -d ' ')" ] && { G=""; sleep 120; }
done
GPUS=($G)
echo "[small] GPUs $G $(date -u)"
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --batch 4 --sig 0 --pf 0 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --side_apart --save_every 1000 --keep --data data/datasets/c1 --shards 1"
run() {   # $1 episodes, $2 steps, $3 gpu
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$3 $L -m lsim.train.cloth_train $NET --episodes $1 \
    --steps $2 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fit_$1 > outputs/lsim/cloth_fit_$1.log 2>&1
  echo "[small] $1 episodes trained ($?) $(date -u)"
  CUDA_VISIBLE_DEVICES=$3 $L scripts/l2_small_fit.py --run cloth_fit_$1 --episodes $1 > outputs/lsim/small_fit_$1.log 2>&1
  echo "[small] $1 episodes scored ($?) $(date -u)"
}
if [ ${#GPUS[@]} -ge 2 ]; then
  run 1 4000 ${GPUS[0]} &
  run 16 8000 ${GPUS[1]} &
  wait
else
  run 1 4000 ${GPUS[0]}
  run 16 8000 ${GPUS[0]}
fi
echo "[small] all done $(date -u)"
