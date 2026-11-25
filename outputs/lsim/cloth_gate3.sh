#!/bin/bash
# The explicit solver again, all checks, after its membrane damping became objective (strain rate, blind to rotation);
# VBD's and Style3D's swing again for the per-frame energy record.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l0_cloth_gate.py stretch swing freefall airfall slide bend drape fold grasp throughput --solver semi
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l0_cloth_gate.py swing --solver vbd,style3d
echo "[cloth gate 3] done $(date -u)"
