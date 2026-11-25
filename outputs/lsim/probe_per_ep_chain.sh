#!/bin/bash
# Episodes or samples (2026-11-20; l1_local_probe_scale.py --per_ep_test): a 3x denser draw from the same episodes.
# Extraction on CPU, then one fully empty GPU (no one's process on it, checked twice a minute apart), ~20 min.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
C=outputs/lsim/probe_scale_ep450
CUDA_VISIBLE_DEVICES="" $L scripts/l1_local_probe_scale.py --extract --per_ep 450 --store_R 4 --cache $C > outputs/lsim/probe_per_ep_extract.log 2>&1 || { echo "[per_ep] extract failed"; exit 1; }
echo "[per_ep] extracted $(date -u)"
empty_gpus() {
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read i u; do echo "$busy" | grep -q "$u" || echo $i; done
}
G=""
while [ -z "$G" ]; do
  for g in $(empty_gpus); do
    sleep 60
    if empty_gpus | grep -qx "$g"; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
echo "[per_ep] GPU $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L scripts/l1_local_probe_scale.py --per_ep_test --cache $C > outputs/lsim/probe_per_ep.log 2>&1 || { echo "[per_ep] failed"; exit 1; }
echo "[per_ep] done $(date -u)"
