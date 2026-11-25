#!/bin/bash
# The local stencil as input (2026-11-19): a linear map of each vertex's 9x9 grid stencil (3 frames) predicts free
# cloth's next frame at 0.80-0.82 of constant velocity's error and a small MLP in the vertex's own frame at 0.73, where
# the 64M network sits at 0.86-0.91 on the same kind of vertices: the network does not get from attention what a local
# patch gives directly. ck1's recipe and data + --patch 4 (the stencil in each vertex's own frame, a force head in it),
# 8,000 steps; scored on the same c1v_test windows as ck1 (paired) and on c1m_test (the probe's vertex subset).
# One fully empty GPU (no one's process on it, checked twice a minute apart).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  for g in $(empty_gpus); do
    sleep 60
    if empty_gpus | grep -qx "$g"; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
echo "[patch] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L -m lsim.train.cloth_train --data data/datasets/c1 --steps 8000 --batch 4 --d 512 --layers 16 --arch qwen --heads 8 \
  --kv_heads 4 --mlp 1536 --lr 6e-4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 --save_every 2000 \
  --keep --reload 1000 --patch 4 --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_ck1_patch4 \
  > outputs/lsim/cloth_ck1_patch4.log 2>&1 || { echo "[patch] training failed"; exit 1; }
echo "[patch] trained $(date -u)"
for s in 4000 7999; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_patch4/step$s.pt --data c1v_test --n 300 --tag patch4_$s \
    > outputs/lsim/where_patch4_$s.log 2>&1
done
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_patch4/step7999.pt --data c1m_test --n 300 --tag patch4_on_c1m \
  > outputs/lsim/where_patch4_on_c1m.log 2>&1
echo "[patch] scored $(date -u)"
