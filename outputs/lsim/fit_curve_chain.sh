#!/bin/bash
# Train / held-out fit curves (scripts/l2_fit_curve.py) for the 64M and 509M cloth runs and the 64M mixed run, packed on
# ONE GPU that no other process uses (2026-11-06: every card carried another user's jobs; wait for an empty one).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
while true; do
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  g=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done | head -1)
  [ -n "$g" ] && break
  sleep 300
done
echo "[fit] GPU $g free $(date -u)"
export CUDA_VISIBLE_DEVICES=$g
$L scripts/l2_fit_curve.py --run cloth_q64 --train c1 --test c1_test > outputs/lsim/fit_curve_q64.log 2>&1 &
$L scripts/l2_fit_curve.py --run cloth_q509 --train c1 --test c1_test > outputs/lsim/fit_curve_q509.log 2>&1 &
$L scripts/l2_fit_curve.py --run cloth_m64 --train c1,r1,s1 --test c1_test,r1_test,s1_test > outputs/lsim/fit_curve_m64.log 2>&1 &
wait
echo "[fit] done $(date -u)"
