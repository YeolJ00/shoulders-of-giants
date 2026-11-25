#!/bin/bash
# Render four c1 episodes (shard 2) and tile them, GPU 3 (next to generation).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
for e in 6 8 9 13; do
    rm -rf outputs/lsim/render_c1/shard_00002_e$e/teacher
    CUDA_VISIBLE_DEVICES=3 /home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender --background --python scripts/blender_render.py -- outputs/lsim/render_c1/shard_00002_e$e 640 480 32 2>&1 | grep -i "error\|\[blender\]\|Traceback"
done
/home/jooyeolyun/anaconda3/envs/lsim/bin/python scripts/l1_render_cloth.py compose data/datasets/c1/shard_00002 6 8 9 13
echo "[c1 render] done $(date -u)"
