#!/bin/bash
# 8 frames per call against 1 frame per call, on cloth (2026-11-06; the user: "queue the chunk comparison"). Does
# predicting the next 8 frames from one network call (cloth_train --chunk 8: the past 8 frames true, the coming ones'
# cloth positions unseen, the exact layer integrating the 8 force frames in order) fit the teacher's data better than
# one call per frame? Two matched 64M runs from scratch on c1 (cloth_q64's recipe: d 512, 16 layers, lr 6e-4, batch 4,
# pf 4, edge 1), 8,000 steps each on ONE GPU each, checkpoints every 2,000; then scripts/l2_chunk_fit.py scores every
# checkpoint at coming frames k = 1..8 (position and shape error, training and held-out windows).
# GPUs: only a card with NO compute process at all (the user's rule: never a GPU another person's job occupies); the
# emptiness is checked twice a minute apart, and a card this chain just launched on is never reused for the other run.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {   # indices of GPUs with no compute process
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
take_gpu() {     # $1: a GPU index to skip; prints a GPU empty on two checks a minute apart
  while true; do
    for g in $(empty_gpus); do
      [ "$g" = "$1" ] && continue
      sleep 60
      empty_gpus | grep -qx "$g" && { echo $g; return; }
    done
    sleep 300
  done
}
R="--data data/datasets/c1 --steps 8000 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 --kv_heads 4 --mlp 1536 \
   --lr 6e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 --save_every 2000 --keep --reload 1000"
g1=$(take_gpu none)
echo "[chunk] per-frame run on GPU $g1 $(date -u)"
CUDA_VISIBLE_DEVICES=$g1 $L -m lsim.train.cloth_train $R --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_ck1 > outputs/lsim/cloth_ck1.log 2>&1 &
p1=$!
g8=$(take_gpu $g1)
echo "[chunk] chunk-8 run on GPU $g8 $(date -u)"
CUDA_VISIBLE_DEVICES=$g8 $L -m lsim.train.cloth_train $R --val "data/datasets/c1_test/shard_*.npz" --chunk 8 --out outputs/lsim/cloth_ck8 > outputs/lsim/cloth_ck8.log 2>&1 &
p8=$!
wait $p1 $p8
echo "[chunk] training done $(date -u)"
gs=$(take_gpu none)
echo "[chunk] scoring on GPU $gs $(date -u)"
CUDA_VISIBLE_DEVICES=$gs $L scripts/l2_chunk_fit.py --runs cloth_ck1,cloth_ck8 > outputs/lsim/chunk_fit.log 2>&1
echo "[chunk] done $(date -u)"
