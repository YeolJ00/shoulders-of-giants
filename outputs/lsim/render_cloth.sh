#!/bin/bash
# The cloth gate scenes: simulate (each solver), render (Blender, OptiX on GPU 3), compose side by side.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
echo "[cloth render] export $(date -u)"
CUDA_VISIBLE_DEVICES=3 /home/jooyeolyun/anaconda3/envs/lsim/bin/python -u scripts/l0_cloth_render.py export "$@" || exit 1
for s in "$@"; do
    echo "[cloth render] blender $s $(date -u)"
    CUDA_VISIBLE_DEVICES=3 /home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender --background --python scripts/blender_render.py -- outputs/lsim/render_cloth/$s 640 480 48 2>&1 | grep -i "error\|\[blender\]\|Traceback"
done
/home/jooyeolyun/anaconda3/envs/lsim/bin/python -u scripts/l0_cloth_render.py compose "$@"
echo "[cloth render] done $(date -u)"
