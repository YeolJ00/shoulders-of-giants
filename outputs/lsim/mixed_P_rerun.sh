#!/bin/bash
# Rerun (2026-11-05) of the mixed restoration fine-tune (train_restore_mixed.sh), which OOMed in pairs_within on
# 2026-11-03 (now chunked): train on GPU 4; shape eval + the 9 clips at steps 1000 and 1999 on GPU 6.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=4 $L -m lsim.train.cloth_train --data data/datasets/c1,data/datasets/r1,data/datasets/s1 --data_w 0.5,0.25,0.25 \
  --steps 2000 --batch 4 --d 512 --layers 16 --arch qwen3d --heads 8 --kv_heads 4 --mlp 1536 --lr 3e-5 --sig 1e-3 --pf 4 \
  --prob_slots --unc --accel --layer 2 --edge 10 --noise_rel \
  --val 'data/datasets/c1_test/shard_*.npz,data/datasets/r1_test/shard_*.npz,data/datasets/s1_test/shard_*.npz' \
  --init outputs/lsim/cloth_m64/step23999.pt --roll 30 --roll_w 1 --roll_n 2 --roll_every 10 --save_every 500 --log_every 50 \
  --keep --restore 1 --out outputs/lsim/cloth_m64P > outputs/lsim/cloth_m64P.log 2>&1 &
for s in 1000 1999; do
  f=outputs/lsim/cloth_m64P/step$s.pt
  until [ -f $f ] || grep -q Traceback outputs/lsim/cloth_m64P.log; do sleep 120; done
  [ -f $f ] || break; sleep 30
  CUDA_VISIBLE_DEVICES=6 $L scripts/l2_shape_long.py --ckpt $f --tag m64P_step$s > outputs/lsim/shape_long_m64P_step$s.log 2>&1
  bash outputs/lsim/render_mixed_P.sh $f outputs/lsim/render_mixed_P$s 6
  echo "[mixed] step $s $(date -u)"
done
wait
