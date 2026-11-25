#!/bin/bash
# The local rule's learning curve over c1's episodes (2026-11-20; scripts/l1_local_probe_scale.py). Extraction on CPU,
# then one fully empty GPU (no one's process on it, checked twice a minute apart) for the fits, ~30-60 min.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
CUDA_VISIBLE_DEVICES="" $L scripts/l1_local_probe_scale.py --extract > outputs/lsim/probe_scale_extract.log 2>&1 || { echo "[scale] extract failed"; exit 1; }
echo "[scale] extracted $(date -u)"
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
echo "[scale] GPU $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L scripts/l1_local_probe_scale.py --fit > outputs/lsim/probe_scale_fit.log 2>&1 || { echo "[scale] fit failed"; exit 1; }
echo "[scale] done $(date -u)"
