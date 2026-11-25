#!/bin/bash
# When the 509M cloth run (q509) ends: GPUs 0,1 -> the 64M mixed-material run (cloth + rope + soft, qwen3d);
# GPU 4 -> r5's real fine-tune from q509's last checkpoint (rq509); GPU 5 -> q509 on PGND, then rq509's checkpoints.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 2371268 2>/dev/null; do sleep 60; done
sleep 30
F=outputs/lsim/cloth_q509/step11999.pt
[ -f $F ] || F=outputs/lsim/cloth_q509/last.pt
echo "[a] q509 ended $(date -u); using $F"
CUDA_VISIBLE_DEVICES=0,1 /home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun --nproc_per_node 2 --master_port 29521 -m lsim.train.cloth_train \
  --data data/datasets/c1,data/datasets/r1,data/datasets/s1 --data_w 0.5,0.25,0.25 \
  --arch qwen3d --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --noise_rel --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 \
  --save_every 2000 --keep \
  --val "data/datasets/c1_test/shard_*.npz,data/datasets/r1_test/shard_*.npz,data/datasets/s1_test/shard_*.npz|outputs/lsim/soft_pilot/shard_*.npz" \
  --out outputs/lsim/cloth_m64 > outputs/lsim/cloth_m64.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l2_real_cloth_finetune.py --init $F --out outputs/lsim/cloth_rq509 --steps 1000 --lr 5e-5 \
  --trunc 10 --teacher_w 1 --back_w 1 --strain_w 1 > outputs/lsim/cloth_rq509.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $F --tag q509 > outputs/lsim/pgnd_eval_q509.log 2>&1
echo "[a] q509 PGND evaluated $(date -u)"
for s in 250 500 750 1000; do
  f=outputs/lsim/cloth_rq509/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_rq509.log; do sleep 60; done
  [ -f $f ] || break; sleep 30
  CUDA_VISIBLE_DEVICES=5 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag rq509_step$s > outputs/lsim/pgnd_eval_rq509_step$s.log 2>&1
  echo "[a] rq509 step $s evaluated $(date -u)"
done
wait
