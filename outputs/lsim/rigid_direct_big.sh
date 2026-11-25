#!/bin/bash
# The third lever for the plain transformer on rigid objects: size (2026-11-23). rigid_pilot128p_attn_direct (1.6M,
# d128 x 6) scores 0.571 on held-out episodes and 0.540 on its own -- no gap, so the fit is limited by training
# length (rigid_direct_levers.sh long), data (data) or the network's size (here). Same recipe, episodes and steps,
# the network 5x larger:
#   rigid_pilot128p_attn_direct_big   --d 256 --layers 8 (8.4M)
# PREDICTION (written before the run): own 0.45-0.52, held-out 0.52-0.57 -- level with the small network on unseen
# episodes (within 0.03), as 8x the parameters was for cloth at equal windows, and as the 8.4M attention trunk
# memorised one episode no faster than the 1.6M. READ: held-out gain >= 0.03 = size is a lever at 128 episodes
# ("a bigger transformer" is worth its cost here); own falls and held-out does not = it memorises, data is next.
#   $1 = GPU (empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1; n=rigid_pilot128p_attn_direct_big
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[big] $n GPU $g $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py --data data/datasets/v0 --episodes 128 --steps 12000 --pair_norm --coulomb --pf 6 \
  --noise 0.003,0.015 --mix --plain --trunk attn --d 256 --layers 8 --direct --out outputs/lsim/$n.pt > outputs/lsim/$n.log 2>&1
echo "[big] $n trained ($?) $(date -u)"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --tag own > outputs/lsim/small_fit_${n}_own.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 32 --shard 1 --tag held > outputs/lsim/small_fit_${n}_held.log 2>&1 &
wait
echo "[big] $n scored $(date -u)"
