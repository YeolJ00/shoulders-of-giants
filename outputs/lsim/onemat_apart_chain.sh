#!/bin/bash
# The one-material test, repeated without its confound (2026-11-21). In both one-material runs (ck1_onemat, 8,000 steps
# from scratch; fp_onemat, 4,000 steps from the continuation) the slot posterior's variance collapsed (every slot is the
# same value: slot NLL -> -5, its gradient norm 100-460 against ~5 on mixed materials), and the one shared
# clip_grad_norm_(.., 1.0) scaled the position signal down by that factor while the slot NLL steered the trunk: fp_onemat
# got WORSE on its own training windows (0.738 vs the start's 0.651) -- so neither run measured whether one material
# can be fit. --side_apart (the slot head reads the trunk detached; side heads clipped apart) removes it. Two 4,000-step
# fine-tunes from the continuation's last weights (one-cycle lr 1e-4, as fit_probe_chain), 1 GPU each:
#   onemat_sa  c1m (1,024 episodes at one material), --side_apart
#   ctl_sa     c1, --side_apart -- what --side_apart alone does on mixed materials (fp_ctl is the coupled control)
# Scored on 300 c1_test (probe R 6), 300 c1m_test and the training windows; paired against fp_ctl / q64_cont.
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
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -2 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 2 ] && G="$both" || sleep 120
done
set -- $G
echo "[sa] onemat_sa on $1, ctl_sa on $2 $(date -u)"
INIT=outputs/lsim/cloth_q64_cont/step23999.pt
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 4000 --lr 1e-4 --batch 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --sig 1e-3 --pf 4 --side_apart --save_every 2000 --keep --init $INIT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 $L -m lsim.train.cloth_train $NET --data data/datasets/c1m \
  --val "data/datasets/c1m_test/shard_*.npz" --out outputs/lsim/cloth_fp_onemat_sa > outputs/lsim/cloth_fp_onemat_sa.log 2>&1 &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$2 $L -m lsim.train.cloth_train $NET --data data/datasets/c1 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fp_ctl_sa > outputs/lsim/cloth_fp_ctl_sa.log 2>&1 &
wait
echo "[sa] trained $(date -u)"
export CUDA_VISIBLE_DEVICES=$1
for r in onemat_sa ctl_sa; do
  for s in 2000 3999; do
    $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$r/step$s.pt --data c1m_test --n 300 \
      --tag fp_${r}_${s}_on_c1m > outputs/lsim/where_fp_${r}_${s}_on_c1m.log 2>&1 &
  done
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$r/step3999.pt --data c1_test --n 300 --probe_R 6 \
    --tag fp_${r}_on_c1_R6 > outputs/lsim/where_fp_${r}_on_c1_R6.log 2>&1 &
done
wait
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_onemat_sa/step3999.pt --data c1m --n 300 \
  --tag fp_onemat_sa_on_train > outputs/lsim/where_fp_onemat_sa_on_train.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_ctl_sa/step3999.pt --data c1_trainsub --n 300 --probe_R 6 \
  --tag fp_ctl_sa_on_train > outputs/lsim/where_fp_ctl_sa_on_train.log 2>&1 &
wait
echo "[sa] scored $(date -u)"
