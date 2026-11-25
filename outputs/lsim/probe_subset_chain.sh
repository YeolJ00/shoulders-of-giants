#!/bin/bash
# Score the one-material network and the all-material one on c1m_test, now also on l1_local_probe's vertex subset
# (2026-11-19). One fully empty GPU (no one's process on it, checked twice a minute apart); ~6 min.
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
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
echo "[subset] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_onemat/step7999.pt --data c1m_test --n 300 --tag onemat_7999 \
  > outputs/lsim/where_onemat_7999.log 2>&1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1m_test --n 300 --tag ck1_on_c1m \
  > outputs/lsim/where_ck1_on_c1m.log 2>&1
echo "[subset] done $(date -u)"
