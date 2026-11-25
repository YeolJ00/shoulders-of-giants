#!/bin/bash
# Fit tests (2026-11-09; the user: "run the overfit and bf16 tests. Use only two GPUs"). Can the 64M cloth network fit
# 100 episodes (shards 0-1 of c1) far below the full-data plateau (one-step 0.68 mm, shape 0.63% held-out)? And does the
# bf16 residual stream set the floor? Two matched runs of the chunk test's per-frame recipe on the same 100 episodes,
# 8,000 steps each, one GPU each: bf16 (default) and --fp32. Then scripts/l2_fit_curve.py scores every checkpoint on
# windows from the same 100 episodes and on held-out c1_test.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
take_gpu() {
  while true; do
    for g in $(empty_gpus); do
      [ "$g" = "$1" ] && continue
      sleep 60
      empty_gpus | grep -qx "$g" && { echo $g; return; }
    done
    sleep 300
  done
}
R="--data data/datasets/c1 --shards 2 --episodes 100 --steps 8000 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 \
   --kv_heads 4 --mlp 1536 --lr 6e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 --save_every 2000 \
   --keep --reload 100000"
ga=$(take_gpu none)
echo "[overfit] bf16 run on GPU $ga $(date -u)"
CUDA_VISIBLE_DEVICES=$ga $L -m lsim.train.cloth_train $R --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_of_bf16 > outputs/lsim/cloth_of_bf16.log 2>&1 &
pa=$!
gb=$(take_gpu $ga)
echo "[overfit] fp32 run on GPU $gb $(date -u)"
CUDA_VISIBLE_DEVICES=$gb $L -m lsim.train.cloth_train $R --val "data/datasets/c1_test/shard_*.npz" --fp32 --out outputs/lsim/cloth_of_fp32 > outputs/lsim/cloth_of_fp32.log 2>&1 &
pb=$!
wait $pa
echo "[overfit] bf16 done $(date -u); scoring it on GPU $ga"
CUDA_VISIBLE_DEVICES=$ga $L scripts/l2_fit_curve.py --run cloth_of_bf16 --train c1 --test c1_test --episodes 100 > outputs/lsim/fit_of_bf16.log 2>&1
wait $pb
echo "[overfit] fp32 done $(date -u); scoring it on GPU $gb"
CUDA_VISIBLE_DEVICES=$gb $L scripts/l2_fit_curve.py --run cloth_of_fp32 --train c1 --test c1_test --episodes 100 > outputs/lsim/fit_of_fp32.log 2>&1
echo "[overfit] all done $(date -u)"
