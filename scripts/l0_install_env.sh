#!/usr/bin/env bash
# Phase 0: create the `lsim` env (Python 3.12) for the learned-simulator track.
# torch cu128 because this box's driver (575.57.08) tops out at CUDA 12.9.
set -euo pipefail
source /home/jooyeolyun/anaconda3/etc/profile.d/conda.sh
if [ ! -x /home/jooyeolyun/anaconda3/envs/lsim/bin/python ]; then
  conda create -y -n lsim python=3.12
fi
PY=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install "newton[sim,importers,remesh]==1.7.0"
$PY -m pip install pymeshlab matplotlib pytest
# PyPI warp-lang >= 1.18 is built for CUDA 13 and needs driver >= 580; this box has 575
# (CUDA 12.9), so swap in the +cu12 build from GitHub releases.
$PY -m pip install --force-reinstall --no-deps \
  "https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0%2Bcu12-py3-none-manylinux_2_28_x86_64.whl"
$PY -m pip freeze > /home/nas5/jooyeolyun/repos/simulation-assestization/requirements-lsim.txt
$PY -m pip list 2>/dev/null | grep -iE "^(torch|newton|warp-lang|mujoco|mujoco-warp|trimesh|coacd|pymeshlab|open3d|usd-core|numpy|scipy) "
echo INSTALL_DONE
