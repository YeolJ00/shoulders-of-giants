#!/bin/bash
# Error by motion family, phase and material terciles (scripts/l2_error_by_kind.py), 2026-11-21: the continued network
# and the side-apart control fine-tune, packed on one empty GPU (checked twice, a minute apart).
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
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
echo "[kind] on $G $(date -u)"
CUDA_VISIBLE_DEVICES=$G $L scripts/l2_error_by_kind.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --tag q64_cont > outputs/lsim/by_kind_q64_cont.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G $L scripts/l2_error_by_kind.py --ckpt outputs/lsim/cloth_fp_ctl_sa/step3999.pt --tag fp_ctl_sa > outputs/lsim/by_kind_fp_ctl_sa.log 2>&1 &
wait
echo "[kind] done $(date -u)"
