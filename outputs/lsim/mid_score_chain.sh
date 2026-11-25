#!/bin/bash
# Mid-run read of the two full-compute runs (2026-11-20): the stencil run at steps 8,000 and 12,000 against q64 at the
# same steps of the same 24k schedule, and the continuation at its step 12,000, all on the same 300 c1_test windows
# (probe R 6) as q64's curve. The trainer's own val is 64 windows and swings +-0.1 mm. One fully empty GPU (no one's
# process on it, checked twice a minute apart); the full chain later rescores step 8,000 under the same tag.
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
echo "[mid] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
for rs in q64_patch4:8000 q64_patch4:12000 q64_cont:12000; do
  r=${rs%%:*}; s=${rs##*:}
  $L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_$r/step$s.pt --data c1_test --n 300 --probe_R 6 \
    --tag ${r}_${s}_on_c1_R6 > outputs/lsim/where_${r}_${s}_on_c1_R6.log 2>&1 &
done
wait
echo "[mid] done $(date -u)"
