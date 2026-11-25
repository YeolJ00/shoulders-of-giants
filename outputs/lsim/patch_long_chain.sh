#!/bin/bash
# Does the stencil input's one-step gain (free cloth 0.76 vs ck1 0.84, paired -0.077 [-0.108, -0.049]) carry over
# seconds? (2026-11-20) Both 8,000-step networks rolled out on the same 256 c1_test episodes (paired over episodes), at
# steps 6,000 and 7,999 (never on one checkpoint); and the stencil network on the local rule's vertex set (c1_test,
# 13x13 clear), against ck1's 0.817 and the local MLP's 0.68 there. All on one fully empty GPU (no one's process on it,
# checked twice a minute apart).
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
echo "[plong] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_patch4/step7999.pt --data c1_test --n 300 --probe_R 6 \
  --tag patch4_on_c1_R6 > outputs/lsim/where_patch4_on_c1_R6.log 2>&1 &
for s in 6000 7999; do
  $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_ck1/step$s.pt --n 256 --tag ck1_step${s}_n256 \
    > outputs/lsim/long_ck1_step${s}_n256.log 2>&1 &
  $L scripts/l2_cloth_long.py --ckpt outputs/lsim/cloth_ck1_patch4/step$s.pt --n 256 --tag patch4_step${s}_n256 \
    > outputs/lsim/long_patch4_step${s}_n256.log 2>&1 &
done
wait
echo "[plong] done $(date -u)"
