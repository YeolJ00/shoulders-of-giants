#!/bin/bash
# The fit at full compute (2026-11-20). Every 8,000-step / 1-GPU comparison (ck1 and its variants) saw 1/6 of the
# windows cloth_q64 saw (24k steps x 2 GPUs x 4), and q64's free-cloth error / constant velocity's on c1_test fell
# 0.87 (ck1) -> 0.78 / 0.74 / 0.70 / 0.68 / 0.67 at its steps 8k / 12k / 16k / 20k / 24k: the fit is compute-limited.
# The stencil input (--patch 4) at 8,000 steps on 1 GPU equals q64 at 8,000 steps on 2 (paired +0.010 [-0.017, +0.038]).
#   A  cloth_q64_patch4: q64's recipe exactly + --patch 4, 24k steps x 2 GPUs, from scratch -- does the stencil input
#      still help at full compute (paired against q64 step 23999)?
#   B  cloth_q64_cont:   q64 step 23999's weights + another 24k steps x 2 GPUs (new optimiser and schedule) -- does the
#      fit keep improving with twice the compute (fresh windows of the same 20k episodes)?
# Four fully empty GPUs (no one's process on them, checked twice a minute apart); scored on c1_test (probe R 6),
# c1v_test, c1m_test, the training subset c1_trainsub, and over seconds (256 c1_test episodes).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
T=/home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' ')
  sleep 60
  b=$(empty_gpus | tr '\n' ' ')
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -4 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 4 ] && G="$both" || sleep 120
done
set -- $G
GA="$1,$2"; GB="$3,$4"
echo "[full] A on $GA, B on $GB $(date -u)"
COMMON="--data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 24000 --lr 6e-4
  --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --save_every 2000 --keep"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GA $T --nproc_per_node 2 --master_port 29561 \
  -m lsim.train.cloth_train $COMMON --val "data/datasets/c1_test/shard_*.npz" --patch 4 --out outputs/lsim/cloth_q64_patch4 > outputs/lsim/cloth_q64_patch4.log 2>&1 &
pA=$!
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$GB $T --nproc_per_node 2 --master_port 29562 \
  -m lsim.train.cloth_train $COMMON --val "data/datasets/c1_test/shard_*.npz" --init outputs/lsim/cloth_q64/step23999.pt --out outputs/lsim/cloth_q64_cont \
  > outputs/lsim/cloth_q64_cont.log 2>&1 &
pB=$!
wait $pA; echo "[full] A trained ($?) $(date -u)"
wait $pB; echo "[full] B trained ($?) $(date -u)"
export CUDA_VISIBLE_DEVICES=${GA%%,*}
for r in q64_patch4 q64_cont; do
  for s in 8000 16000 23999; do
    [ -f outputs/lsim/cloth_$r/step$s.pt ] || continue
    $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step$s.pt --data c1_test --n 300 --probe_R 6 \
      --tag ${r}_${s}_on_c1_R6 > outputs/lsim/where_${r}_${s}_on_c1_R6.log 2>&1 &
  done
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step23999.pt --data c1v_test --n 300 --tag ${r}_on_c1v \
    > outputs/lsim/where_${r}_on_c1v.log 2>&1 &
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step23999.pt --data c1m_test --n 300 --tag ${r}_on_c1m \
    > outputs/lsim/where_${r}_on_c1m.log 2>&1 &
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step23999.pt --data c1_trainsub --n 300 --probe_R 6 \
    --tag ${r}_on_train > outputs/lsim/where_${r}_on_train.log 2>&1 &
  wait
done
for r in q64 q64_patch4 q64_cont; do
  $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_$r/step23999.pt --n 256 --tag ${r}_step23999_n256 \
    > outputs/lsim/long_${r}_step23999_n256.log 2>&1 &
done
wait
echo "[full] scored $(date -u)"
