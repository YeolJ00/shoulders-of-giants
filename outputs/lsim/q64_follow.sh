#!/bin/bash
# After q64 (64M Qwen-style, one-step recipe; c1_test 27 / 72 / 108 mm at 1 / 2 / 3.8 s, level with v6e):
#   GPU 7: q64L = v6e's long-rollout fine-tune (100 frames, lr 3e-5, 2,000 steps) from q64's last checkpoint
#   GPU 6: q64 on PGND's held-out windows, then rq64 = r5's real fine-tune (two-sided + strain, 1,000 steps, lr 1e-4)
#          from q64; every q64L / rq64 checkpoint scored on GPU 6 (long horizon / PGND).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
I=outputs/lsim/cloth_q64/step23999.pt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=7 $L -u -m lsim.train.cloth_train --data data/datasets/c1 \
  --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 2000 --save_every 500 --keep --batch 4 --lr 3e-5 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1 \
  --init $I --roll 100 --roll_n 2 --roll_every 10 \
  --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/cloth_q64L > outputs/lsim/cloth_q64L.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $I --tag q64 > outputs/lsim/pgnd_eval_q64.log 2>&1
echo "[f] q64 PGND evaluated $(date -u)"
CUDA_VISIBLE_DEVICES=6 $L -u scripts/l2_real_cloth_finetune.py --init $I --out outputs/lsim/cloth_rq64 --steps 1000 --lr 1e-4 \
  --trunc 10 --teacher_w 1 --back_w 1 --strain_w 1 > outputs/lsim/cloth_rq64.log 2>&1 &
wt(){ until [ -f $1 ] || grep -q Traceback $2; do sleep 60; done; [ -f $1 ] && sleep 30; }
for s in 500 1000 1500 1999; do
  f=outputs/lsim/cloth_q64L/step$s.pt; wt $f outputs/lsim/cloth_q64L.log || break
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_cloth_long.py --ckpt $f --tag q64L_step$s > outputs/lsim/long_q64L_step$s.log 2>&1
  echo "[f] q64L step $s evaluated $(date -u)"
  if [ $s = 1000 ]; then for r in 500 1000; do f=outputs/lsim/cloth_rq64/step$r.pt; wt $f outputs/lsim/cloth_rq64.log || continue
    CUDA_VISIBLE_DEVICES=6 $L scripts/l2_real_cloth_pgnd_eval.py --ckpt $f --tag rq64_step$r > outputs/lsim/pgnd_eval_rq64_step$r.log 2>&1
    echo "[f] rq64 step $r evaluated $(date -u)"; done; fi
done
wait
