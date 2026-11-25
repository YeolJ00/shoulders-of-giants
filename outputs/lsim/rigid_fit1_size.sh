#!/bin/bash
# Why the rigid network does not fit one episode (2026-11-23): not the noise (rigid_fit_1_clean, --plain --noise 0,0
# --pf 0: one frame 0.355 of constant velocity, 0.468 over every frame; with the noise 0.358) and only partly the
# augmentation (0.517 with it). Two arms on the clean setting, 6,000 steps, scored by l2_small_fit_rigid.py and
# scratch rigid_frame_err.py:
#   rigid_fit_1_wide     --d 384 --layers 8          size: 2.4M -> ~25M parameters
#   rigid_fit_1_direct   --direct                    no physics layer (baseline B2's head): is it the layer?
#   $1 = GPU (empty twice a minute apart), $2 = name, $3.. = the arm's flags
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=$2; shift 2
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rigids] $n GPU $g: $* $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 1 --steps 6000 --pair_norm --coulomb \
  --mix --noise 0,0 --pf 0 --plain "$@" --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[rigids] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 1 > outputs/lsim/small_fit_$n.log 2>&1
echo "[rigids] $n scored ($?) $(date -u)"
