#!/bin/bash
# Render four dev cloth episodes and tile them.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
for e in 3 8 14 4; do
    CUDA_VISIBLE_DEVICES=3 /home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender --background --python scripts/blender_render.py -- outputs/lsim/render_c1/shard_00000_e$e 640 480 32 2>&1 | grep -i "error\|\[blender\]\|Traceback"
done
/home/jooyeolyun/anaconda3/envs/lsim/bin/python scripts/l1_render_cloth.py compose data/datasets/c1_dev/shard_00000 3 8 14 4
echo "[c1 dev render] done $(date -u)"
