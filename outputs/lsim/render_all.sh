#!/bin/bash
# Render every exported clip on GPU 3 (packed beside the v1/B2 evaluations; Blender holds ~2 GB), then
# compose the side-by-side videos. Resumable: frames on disk are skipped.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
B=/home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
echo "[render] start $(date -u)"
for d in outputs/lsim/render/*/; do
    CUDA_VISIBLE_DEVICES=3 $B --background --python scripts/blender_render.py -- $d 640 480 48 2>&1 | grep "^\[blender\]\|Error\|Traceback"
done
$L -u scripts/l2_render.py compose 2>&1 | grep "^\[compose\]\|Error\|Traceback"
echo "[render] done $(date -u)"
