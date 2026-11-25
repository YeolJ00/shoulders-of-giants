#!/bin/bash
# The frame-rate study's model test (2026-11-18; the user: test the 30 fps hypothesis). The same 1,024 + 128 episodes at
# 30 / 60 / 120 fps (l1_generate_cloth.py --sub 4, seeds 5 / 6), one matched 64M Qwen run per rate from scratch (ck1's
# recipe, 8,000 steps, windows of 8 + 8 frames AT THAT RATE; --sig = 1e-3 (30 / fps)^2: the noise the same in force
# units, as the position and edge losses are), then scripts/l2_frame_rate_fit.py on each and its merge.
# GPUs: three at most, only cards with no compute process on two checks a minute apart.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
take_gpu() {     # $1: GPU indices already taken (space-separated); prints one empty on two checks a minute apart
  while true; do
    for g in $(empty_gpus); do
      echo " $1 " | grep -q " $g " && continue
      sleep 60
      empty_gpus | grep -qx "$g" && { echo $g; return; }
    done
    sleep 300
  done
}
until grep -q "shard 16/16" outputs/lsim/gen_c1_fps.log; do sleep 30; done
sleep 20
for r in 30 60 120; do [ -d data/datasets/c1_test_fps$r ] && mv data/datasets/c1_test_fps$r data/datasets/c1_fps${r}_test; done
R="--steps 8000 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --pf 4 --prob_slots \
   --unc --accel --layer 2 --edge 1 --save_every 2000 --keep --reload 0"
taken=""
declare -A gpu pid
for r in 120 60 30; do
  case $r in 30) sig=1e-3;; 60) sig=2.5e-4;; 120) sig=6.25e-5;; esac
  g=$(take_gpu "$taken"); taken="$taken $g"; gpu[$r]=$g
  echo "[fps] $r fps run on GPU $g $(date -u)"
  LSIM_FPS=$r CUDA_VISIBLE_DEVICES=$g $L -m lsim.train.cloth_train $R --fps $r --sig $sig --data data/datasets/c1_fps$r \
    --val "data/datasets/c1_fps${r}_test/shard_*.npz" --out outputs/lsim/cloth_fps$r > outputs/lsim/cloth_fps$r.log 2>&1 &
  pid[$r]=$!
done
for r in 30 60 120; do
  ( wait ${pid[$r]} 2>/dev/null; while kill -0 ${pid[$r]} 2>/dev/null; do sleep 30; done
    echo "[fps] $r fps trained $(date -u); scoring on GPU ${gpu[$r]}"
    LSIM_FPS=$r CUDA_VISIBLE_DEVICES=${gpu[$r]} $L scripts/l2_frame_rate_fit.py --run cloth_fps$r > outputs/lsim/fit_fps$r.log 2>&1
    echo "[fps] $r fps scored $(date -u)" ) &
done
wait
$L scripts/l2_frame_rate_fit.py --merge > outputs/lsim/fit_fps_merge.log 2>&1
echo "[fps] all done $(date -u)"
