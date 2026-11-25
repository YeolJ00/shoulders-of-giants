#!/bin/bash
# Does the finer time step's gain survive more training? (2026-11-21) The frame-rate model test (frame_rate_chain.sh,
# 2026-11-18: the same 1,024 episodes at 30 / 60 / 120 fps, 8,000 steps each on 1 GPU) was read per step ("explains
# only 15-20% of a step's acceleration at every rate: not what stops the fit"), but at the same physical time the
# 120 fps network was a third better: after 1/30 s 1.11 vs 1.66 mm (30 fps constant velocity 1.91; paired -0.55
# [-0.68, -0.44]), i.e. ~0.58 vs ~0.87 of constant velocity -- where every 8,000-step / 1-GPU run sat before the
# compute correction, and the fully trained 30 fps network reaches ~0.63. Continue both from their step 7999 for
# 16,000 more steps (new optimiser, one-cycle lr 6e-4 -- as the continuation did), 1 GPU each, the same recipe and
# data; then the same scorer, on the held-out windows and on training windows (a gap at 1,024 episodes?).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' ')
  sleep 60
  b=$(empty_gpus | tr '\n' ' ')
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | tail -2 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 2 ] && G="$both" || sleep 120
done
set -- $G
echo "[fpsc] 120 fps on $1, 30 fps on $2 $(date -u)"
R="--steps 16000 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 --kv_heads 4 --mlp 1536 --lr 6e-4 --pf 4 --prob_slots \
   --unc --accel --layer 2 --edge 1 --save_every 4000 --keep --reload 0"
LSIM_FPS=120 CUDA_VISIBLE_DEVICES=$1 $L -m lsim.train.cloth_train $R --fps 120 --sig 6.25e-5 --data data/datasets/c1_fps120 \
  --val "data/datasets/c1_fps120_test/shard_*.npz" --init outputs/lsim/cloth_fps120/step7999.pt \
  --out outputs/lsim/cloth_fps120_cont > outputs/lsim/cloth_fps120_cont.log 2>&1 &
p1=$!
LSIM_FPS=30 CUDA_VISIBLE_DEVICES=$2 $L -m lsim.train.cloth_train $R --fps 30 --sig 1e-3 --data data/datasets/c1_fps30 \
  --val "data/datasets/c1_fps30_test/shard_*.npz" --init outputs/lsim/cloth_fps30/step7999.pt \
  --out outputs/lsim/cloth_fps30_cont > outputs/lsim/cloth_fps30_cont.log 2>&1 &
p2=$!
wait $p1; echo "[fpsc] 120 trained ($?) $(date -u)"
wait $p2; echo "[fpsc] 30 trained ($?) $(date -u)"
LSIM_FPS=120 CUDA_VISIBLE_DEVICES=$1 $L scripts/l2_frame_rate_fit.py --run cloth_fps120_cont --tag _cont > outputs/lsim/fit_fps120_cont.log 2>&1 &
LSIM_FPS=30 CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_frame_rate_fit.py --run cloth_fps30_cont --tag _cont > outputs/lsim/fit_fps30_cont.log 2>&1 &
wait
LSIM_FPS=120 CUDA_VISIBLE_DEVICES=$1 $L scripts/l2_frame_rate_fit.py --run cloth_fps120_cont --test data/datasets/c1_fps120 \
  --tag _cont_train > outputs/lsim/fit_fps120_cont_train.log 2>&1 &
LSIM_FPS=30 CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_frame_rate_fit.py --run cloth_fps30_cont --test data/datasets/c1_fps30 \
  --tag _cont_train > outputs/lsim/fit_fps30_cont_train.log 2>&1 &
wait
echo "[fpsc] scored $(date -u)"
