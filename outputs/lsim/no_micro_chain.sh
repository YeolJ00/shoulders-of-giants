#!/bin/bash
# Without the micrometre stretch reading (2026-11-21). The trained network reads the clean frames' 1e-4 edge strains
# through asinh(1e5 strain) (scripts/l2_input_jitter.py: 10 um of jitter moves its prediction by 0.17 of constant
# velocity's error, all through that feature; worth ~0.04 of its accuracy) -- information the teacher does not need
# (from 3 frames jittered 10-30 um it loses nothing) and that neither real data nor the model's own rollouts have (its
# cloth strains 27% after 3.8 s). Does a network denied it fit as well, and does it roll out / transfer better?
# Two 4,000-step fine-tunes from the continuation's last weights, exactly the side-apart control's recipe (fp_ctl_sa:
# one-cycle lr 1e-4, 1 GPU x batch 4, --side_apart), 1 GPU each:
#   nojit   --in_jit 1e-5: the network sees every cloth vertex of every frame jittered 10 um per axis (training only)
#   nofine  --drop_fine: the asinh(1e5 strain) feature zeroed
# Scored against fp_ctl_sa, paired: one frame on 300 c1_test (probe R 6) and the training windows; the same under input
# jitter (0-300 um); 256 c1_test episodes rolled out to 3.8 s; PGND's real cloth before any real fine-tune.
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
echo "[nomicro] nojit on $1, nofine on $2, control scores on $3 $(date -u)"
INIT=outputs/lsim/cloth_q64_cont/step23999.pt
NET="--arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 4000 --lr 1e-4 --batch 4 --prob_slots --unc
  --accel --layer 2 --edge 1.0 --sig 1e-3 --pf 4 --side_apart --save_every 2000 --keep --init $INIT --data data/datasets/c1"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1 $L -m lsim.train.cloth_train $NET --in_jit 1e-5 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fp_nojit > outputs/lsim/cloth_fp_nojit.log 2>&1 &
P1=$!
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$2 $L -m lsim.train.cloth_train $NET --drop_fine \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_fp_nofine > outputs/lsim/cloth_fp_nofine.log 2>&1 &
P2=$!

score() {   # $1 run, $2 gpu: every score but the clean one-frame fit, packed on one card
  local r=$1 g=$2 c=outputs/lsim/cloth_fp_$1/step3999.pt
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_input_jitter.py --ckpt $c --sigmas_um 0,10,30,100,300 --tag fp_$r \
    > outputs/lsim/jitter_fp_$r.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_cloth_long.py --ckpt $c --n 256 --tag fp_${r}_step3999_n256 \
    > outputs/lsim/long_fp_$r.log 2>&1 &
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $c --tag fp_$r > outputs/lsim/pgnd_eval_fp_$r.log 2>&1 &
  wait
}
(score ctl_sa $3; echo "[nomicro] control scored $(date -u)") &
wait $P1 $P2
echo "[nomicro] trained $(date -u)"
for k in "nojit $1" "nofine $2"; do
  set -- $k
  ( CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$1/step3999.pt --data c1_test --n 300 \
      --probe_R 6 --tag fp_${1}_on_c1_R6 > outputs/lsim/where_fp_${1}_on_c1_R6.log 2>&1 &
    CUDA_VISIBLE_DEVICES=$2 $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_$1/step3999.pt --data c1_trainsub \
      --n 300 --probe_R 6 --tag fp_${1}_on_train > outputs/lsim/where_fp_${1}_on_train.log 2>&1 &
    score $1 $2; wait ) &
done
wait
echo "[nomicro] scored $(date -u)"
