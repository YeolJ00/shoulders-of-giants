#!/bin/bash
# The small-fit test on the rigid track (2026-11-22; the user: "Not even accurately predicting simulation test on rigid
# objects?"). Cloth: one episode fitted to 0.044 of constant velocity and replayed to < 1 mm for 2.4 s. Rigid was never
# pushed past October's unit check (one episode, 1,500 steps: 5 mm after 8 frames, 14-24 after 30). One v0 episode
# (5 rigid bodies, a hinged part, the gripper; 6 s), v1's recipe (--pair_norm --coulomb --pf 6 --noise 0.003,0.015
# --mix), 12,000 steps by scripts/l2_overfit.py, scored by scripts/l2_small_fit_rigid.py:
#   rigid_fit_1        the trainer's view as it is (random yaw + shift, hidden slots, random shorter past)
#   rigid_fit_1_plain  --plain (no augmentation, every slot given, every past frame)
# Both on GPU $1 (small networks: 128 wide, 6 layers), once it is empty twice a minute apart.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rigid1] GPU $g $(date -u)"
R="--data data/datasets/v0 --episodes 1 --steps 12000 --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix"
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py $R --out outputs/lsim/rigid_fit_1.pt > outputs/lsim/rigid_fit_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=$g $L scripts/l2_overfit.py $R --plain --out outputs/lsim/rigid_fit_1_plain.pt > outputs/lsim/rigid_fit_1_plain.log 2>&1 &
wait
echo "[rigid1] trained $(date -u)"
for n in rigid_fit_1 rigid_fit_1_plain; do
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/$n.pt --episodes 1 > outputs/lsim/small_fit_$n.log 2>&1
done
echo "[rigid1] scored $(date -u)"
