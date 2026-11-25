#!/bin/bash
# What holds the fully trained network at ~0.63? (2026-11-21) Not data (train = test), size (509M level), length (-0.02
# for twice the windows), the stencil input, nor a material-blind floor (the teacher averaged over 15 other materials:
# 0.74 off surfaces; the networks beat it on all materials and fall short of it at one material). Three 4,000-step
# fine-tunes from the continuation's last weights (new optimiser, one-cycle lr 1e-4), 1 GPU each:
#   ctl     the recipe as is on c1                       -- what more annealing alone buys
#   clean   c1 without the chunk noise or pushforward    -- is 1 mm of input noise smoothing the learned map? (tested
#           only at 8,000 steps, where every network sat at ~0.85)
#   onemat  c1m, every cloth at one material             -- can it learn the map when the material never changes?
#           (ditto: the 8,000-step one-material test predates the compute correction)
# Scored on 300 c1_test windows (probe R 6), 300 c1m_test windows and its own training windows; paired against ctl.
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
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -3 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 3 ] && G="$both" || sleep 120
done
set -- $G
echo "[fp] ctl on $1, clean on $2, onemat on $3 $(date -u)"
INIT=outputs/lsim/cloth_q64_cont/step23999.pt
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 4000 --lr 1e-4 --batch 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --save_every 2000 --keep --init $INIT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 $L -m lsim.train.cloth_train $NET --data data/datasets/c1 \
  --val "data/datasets/c1_test/shard_*.npz" --sig 1e-3 --pf 4 --out outputs/lsim/cloth_fp_ctl > outputs/lsim/cloth_fp_ctl.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$2 $L -m lsim.train.cloth_train $NET --data data/datasets/c1 \
  --val "data/datasets/c1_test/shard_*.npz" --sig 0 --pf 0 --out outputs/lsim/cloth_fp_clean > outputs/lsim/cloth_fp_clean.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$3 $L -m lsim.train.cloth_train $NET --data data/datasets/c1m \
  --val "data/datasets/c1m_test/shard_*.npz" --sig 1e-3 --pf 4 --out outputs/lsim/cloth_fp_onemat > outputs/lsim/cloth_fp_onemat.log 2>&1 &
wait
echo "[fp] trained $(date -u)"
export CUDA_VISIBLE_DEVICES=$1
for r in ctl clean onemat; do
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$r/step3999.pt --data c1_test --n 300 --probe_R 6 \
    --tag fp_${r}_on_c1_R6 > outputs/lsim/where_fp_${r}_on_c1_R6.log 2>&1 &
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$r/step3999.pt --data c1m_test --n 300 \
    --tag fp_${r}_on_c1m > outputs/lsim/where_fp_${r}_on_c1m.log 2>&1 &
done
wait
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_ctl/step3999.pt --data c1_trainsub --n 300 --probe_R 6 \
  --tag fp_ctl_on_train > outputs/lsim/where_fp_ctl_on_train.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_clean/step3999.pt --data c1_trainsub --n 300 --probe_R 6 \
  --tag fp_clean_on_train > outputs/lsim/where_fp_clean_on_train.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_onemat/step3999.pt --data c1m --n 300 \
  --tag fp_onemat_on_train > outputs/lsim/where_fp_onemat_on_train.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --data c1m --n 300 \
  --tag q64_cont_on_c1m_train > outputs/lsim/where_q64_cont_on_c1m_train.log 2>&1 &
wait
echo "[fp] scored $(date -u)"
