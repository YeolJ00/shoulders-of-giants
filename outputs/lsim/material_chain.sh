#!/bin/bash
# Does the one-frame map need the material, and does the network use it (2026-11-19)? One fully empty GPU (no one's
# process on it, checked twice a minute apart), light jobs packed:
#   the teacher restarted from its full state with the material changed (l1_teacher_sensitivity), and
#   both networks (positions only, + true velocity) scored with the material given / hidden / another's (l2_error_where --material)
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
echo "[material] GPU $G $(date -u)"
export CUDA_VISIBLE_DEVICES=$G
$L scripts/l1_teacher_sensitivity.py > outputs/lsim/teacher_sensitivity.log 2>&1 &
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1/step7999.pt --data c1v_test --n 300 --tag ck1_on_c1v_mat \
  --material > outputs/lsim/where_ck1_on_c1v_mat.log 2>&1
$L scripts/l2_error_where.py --ckpt outputs/lsim/cloth_ck1_oracle/step7999.pt --data c1v_test --n 300 \
  --tag oracle_v_7999_mat --material > outputs/lsim/where_oracle_v_7999_mat.log 2>&1
echo "[material] networks scored $(date -u)"
wait
echo "[material] done $(date -u)"
