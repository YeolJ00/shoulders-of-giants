#!/bin/bash
# The Qwen-style cloth network (lsim.model.cloth_qwen) at two sizes, v3's recipe otherwise (layer 2, edge loss 1,
# pushforward 4, slot posterior, uncertainty head, accelerations as inputs), both on the same 192k training windows
# (2x v3's 96k): the size comparison is at equal data.
#   q509: 508.9M (d 1024, 32 layers, 16 / 8 heads, MLP 3072), 4 GPUs x 4 windows, 12k steps, lr 3e-4, ~37 h
#   q64:   64.3M (d 512, 16 layers, 8 / 4 heads, MLP 1536),   2 GPUs x 4 windows, 24k steps, lr 6e-4, ~15 h
# One GPU, batch 4: 10.9 s / step (509M, peak 25 GiB) and 2.25 s / step (64M, 8 GiB) after the optimisations in
# cloth_qwen (flat projections, per-sample flash attention, matmul time attention, compiled elementwise).
#   train_cloth_q.sh <tag> <gpus, comma-separated> <port> <steps> <lr> <d> <layers> <heads> <kv_heads> <mlp>
cd /home/nas5/jooyeolyun/repos/simulation-assestization
n=$(echo $2 | tr ',' '\n' | wc -l)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$2 /home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun \
  --nproc_per_node $n --master_port $3 -m lsim.train.cloth_train --data data/datasets/c1 \
  --arch qwen --d $6 --layers $7 --heads $8 --kv_heads $9 --mlp ${10} \
  --steps $4 --lr $5 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --save_every 2000 --keep --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_$1 > outputs/lsim/cloth_$1.log 2>&1
