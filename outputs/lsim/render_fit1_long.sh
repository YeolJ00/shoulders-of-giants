#!/bin/bash
# Video of the one-episode --own_up network trained 12,000 steps replaying its episode (2026-11-22): export on GPU $1,
# Blender, compose. outputs/lsim/render_small_fit/shard_00000_e0_t7_fit1_local12k_replay.mp4
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
D=outputs/lsim/render_small_fit/shard_00000_e0_t7_fit1_local12k_replay
CUDA_VISIBLE_DEVICES=$1 $L scripts/l2_render_cloth.py export outputs/lsim/cloth_fit_1_ownup_pf12k/step11999.pt \
  data/datasets/c1/shard_00000 0 7 end fit1_local12k_replay --replay --out outputs/lsim/render_small_fit
echo "[render] exported ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$1 $B --background --python scripts/blender_render.py -- $D > $D.blender.log 2>&1
echo "[render] blender ($?) $(date -u)"
$L scripts/l2_render_cloth.py compose $D --out outputs/lsim/render_small_fit
echo "[render] composed ($?) $(date -u)"
