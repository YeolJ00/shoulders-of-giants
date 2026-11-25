#!/bin/bash
# The cloth gate's remaining checks: the explicit solver's drape (hang criterion), air, fold, grasp and throughput;
# VBD's and Style3D's drape and throughput.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l0_cloth_gate.py drape airfall fold grasp throughput --solver semi
CUDA_VISIBLE_DEVICES=4 $L -u scripts/l0_cloth_gate.py drape throughput --solver vbd,style3d
echo "[cloth gate 2] done $(date -u)"
