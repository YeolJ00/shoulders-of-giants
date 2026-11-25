#!/bin/bash
# The page's four episodes again, from the regenerated c1 (the same scenes and materials as before: same seeds), GPU 4.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
mkdir -p outputs/lsim/render_c1_old && mv outputs/lsim/render_c1/shard_00002_* outputs/lsim/render_c1_old/ 2>/dev/null
$L scripts/l1_render_cloth.py export data/datasets/c1/shard_00002 6 8 9 13
for e in 6 8 9 13; do
    CUDA_VISIBLE_DEVICES=4 /home/nas5/jaeseonglee/blender-4.4.3-linux-x64/blender --background --python scripts/blender_render.py -- outputs/lsim/render_c1/shard_00002_e$e 640 480 32 2>&1 | grep -i "error\|\[blender\]\|Traceback"
done
$L scripts/l1_render_cloth.py compose data/datasets/c1/shard_00002 6 8 9 13
echo "[c1 render] done $(date -u)"
