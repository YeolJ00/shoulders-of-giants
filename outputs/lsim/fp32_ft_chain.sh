#!/bin/bash
# Precision at full compute (2026-11-21). The only training test of bf16 ran at 8,000 steps on 100 episodes (cloth_of_*:
# 1.85 -> 1.72 mm on its training windows, -7%, read as "precision is not the floor") where every network sat at
# ~0.85-0.9 of constant velocity; the fully trained network sits at ~0.65 and nothing tried moves it. The residual
# stream is bf16 (8-bit mantissa: 0.4% of an activation), the teacher's one-frame forces come from strains of ~1e-4
# (micrometres on a centimetre edge). Fine-tune the continuation's last weights for 4,000 steps in fp32 throughout (no
# autocast; matmuls fp32, PyTorch's default no-TF32), else exactly fit_probe_chain's control (fp_ctl: 1 GPU x batch 4,
# one-cycle lr 1e-4): here 2 GPUs x batch 2 (fp32 is ~4x slower), the same 4 windows per step.
# Scored on 300 c1_test (probe R 6), 300 c1m_test and the c1 training windows; paired against fp_ctl.
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
  both=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -2 | tr '\n' ' ')
  [ $(echo $both | wc -w) -ge 2 ] && G="$both" || sleep 120
done
set -- $G
echo "[fp32] on $1,$2 $(date -u)"
INIT=outputs/lsim/cloth_q64_cont/step23999.pt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$1,$2 $T --nproc_per_node 2 --master_port 29571 \
  -m lsim.train.cloth_train --data data/datasets/c1 --val "data/datasets/c1_test/shard_*.npz" --arch qwen --d 512 \
  --layers 16 --heads 8 --kv_heads 4 --mlp 1536 --steps 4000 --lr 1e-4 --batch 2 --sig 1e-3 --pf 4 --prob_slots --unc \
  --accel --layer 2 --edge 1.0 --fp32 --save_every 2000 --keep --init $INIT --out outputs/lsim/cloth_fp_fp32 \
  > outputs/lsim/cloth_fp_fp32.log 2>&1
echo "[fp32] trained ($?) $(date -u)"
export CUDA_VISIBLE_DEVICES=$1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_fp32/step3999.pt --data c1_test --n 300 --probe_R 6 \
  --tag fp_fp32_on_c1_R6 > outputs/lsim/where_fp_fp32_on_c1_R6.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_fp32/step3999.pt --data c1m_test --n 300 \
  --tag fp_fp32_on_c1m > outputs/lsim/where_fp_fp32_on_c1m.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_fp32/step3999.pt --data c1_trainsub --n 300 --probe_R 6 \
  --tag fp_fp32_on_train > outputs/lsim/where_fp_fp32_on_train.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_fp_fp32/step2000.pt --data c1_test --n 300 --probe_R 6 \
  --tag fp_fp32_2000_on_c1_R6 > outputs/lsim/where_fp_fp32_2000_on_c1_R6.log 2>&1 &
wait
echo "[fp32] scored $(date -u)"
