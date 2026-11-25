#!/bin/bash
# The fit question with the candidate network (2026-11-22): q64's recipe exactly (all 20,032 c1 episodes, 24,000 steps
# x 2 GPUs x batch 4 = 192k windows) + --side_apart + $3.. (the frame flags: --own_up). cloth_q64 on c1_test, one
# frame, error / constant velocity: 0.672 at 24k steps (300 windows, free cloth), 0.656 after 48k; by family
# (l2_coverage_score: docs/results/l2_coverage_cov_full_q64_step*.json, cov_full_q64_cont) 0.59-0.78. Evidence for it:
# one episode through the trainer 0.641 -> 0.092 in its own frame; 2,000 episodes at step 3,000 on held-out windows
# 0.949 against the world-frame twin's 1.163 (-0.214 [-0.269, -0.177]). Precedent against: --patch 4 was worth 2x the
# windows early and level at full compute. The read: paired against cloth_q64 at steps 4,000 / 8,000 / 16,000 / 23,999
# on the same windows (--side_apart alone is worth -0.013 to -0.018).
# Launched 2026-11-22 20:30 UTC as cloth_q64_ownup on the evidence at step 6,000 of the 2,000-episode pair: one frame
# 0.866 against 0.973 (-0.106 [-0.132, -0.089]; the lead halved since 3,000) and held-out rollouts on the same 112
# episodes 37 / 105 / 151 mm at 1 s / 2 s / 3.8 s against 65 / 245 / 296 (-28 / -140 / -145, strain 166% vs 289%).
# Prediction on file (DIRECTION.md): ~0.656 at 24k steps, q64 continued's level -- faster, not lower. STOP RULE: stop it
# if the pair's 9,000- or 11,999-step score puts --own_up level or behind on one frame AND its held-out rollouts at
# 11,999 are no better than the twin's (cov_seven_step11999: 42 / 135 / 170 mm, 48 episodes a family).
#   $1 = run name (outputs/lsim/<name>), $2 = "g1,g2" (two GPUs, each checked empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
T=/home/jooyeolyun/anaconda3/envs/lsim/bin/torchrun
n=$1; G=$2; shift 2
busy() {
  for g in ${G//,/ }; do
    u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u" && return 0
  done
  return 1
}
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[full] $n on GPUs $G: $* $(date -u)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$G $T --nproc_per_node 2 --master_port 29571 \
  -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 --heads 8 --kv_heads 4 --mlp 1536 \
  --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel --layer 2 --edge 1.0 --side_apart "$@" \
  --save_every 2000 --keep --val "data/datasets/c1_test/shard_*.npz" --out outputs/lsim/$n > outputs/lsim/$n.log 2>&1
echo "[full] $n trained ($?) $(date -u)"
g1=${G%%,*}
CUDA_VISIBLE_DEVICES=$g1 $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$n/step23999.pt --tag cov_${n}_step23999 \
  > outputs/lsim/coverage_score_${n}.log 2>&1
echo "[full] $n scored ($?) $(date -u)"
