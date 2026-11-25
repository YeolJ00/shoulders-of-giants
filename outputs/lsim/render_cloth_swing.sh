#!/bin/bash
# Re-render the swing with the reframed camera, then compose it.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
CUDA_VISIBLE_DEVICES=3 /home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender --background --python scripts/blender_render.py -- outputs/lsim/render_cloth/swing 640 480 48 2>&1 | grep -i "error\|\[blender\]\|Traceback"
/home/jooyeolyun/anaconda3/envs/lsim/bin/python -u scripts/l0_cloth_render.py compose swing
echo "[swing] done $(date -u)"
