#!/bin/bash
# Mid-run read of the frame-rate continuation (2026-11-21): once both runs have saved step8000.pt (8,000 of their
# 16,000 more steps), score their checkpoints so far with the same scorer (tag _cont_mid), on one empty GPU.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
until [ -f outputs/lsim/cloth_fps120_cont/step8000.pt ] && [ -f outputs/lsim/cloth_fps30_cont/step8000.pt ]; do sleep 60; done
sleep 90
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  a=$(empty_gpus | tr '\n' ' '); sleep 60; b=$(empty_gpus | tr '\n' ' ')
  G=$(comm -12 <(echo $a | tr ' ' '\n' | sort) <(echo $b | tr ' ' '\n' | sort) | sort -n | head -1)
  [ -z "$G" ] && sleep 120
done
echo "[fpsmid] on $G $(date -u)"
LSIM_FPS=120 CUDA_VISIBLE_DEVICES=$G $L scripts/l2_frame_rate_fit.py --run cloth_fps120_cont --tag _cont_mid > outputs/lsim/fit_fps120_cont_mid.log 2>&1 &
LSIM_FPS=30 CUDA_VISIBLE_DEVICES=$G $L scripts/l2_frame_rate_fit.py --run cloth_fps30_cont --tag _cont_mid > outputs/lsim/fit_fps30_cont_mid.log 2>&1 &
wait
echo "[fpsmid] scored $(date -u)"
