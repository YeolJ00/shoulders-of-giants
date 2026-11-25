#!/bin/bash
# Re-score the every-input-given cloth networks with the material GIVEN in rollouts (2026-11-24). l2_coverage_score
# scores one frame with every slot given but rolls out with the material HIDDEN (l2_cloth_long's protocol). The twin
# (`cloth_cov_seven_ownup`) trained with slots hidden half the time; `--given` and `--given --random_past` never saw a
# hidden slot (p_hidden 0), so their rollouts (64 / 127 / 154 and 73 / 149 / 181 mm, area 5x and 11x) were run on an
# input they never trained on -- a scoring mismatch, not a property of the recipe. All three networks again, material
# given in rollouts (`--roll_given`), one GPU, packed (evaluation only):
#   cov_seven_ownup_given_rg, cov_seven_ownup_given_rp_rg, cov_seven_ownup_rg (the twin, for a paired read)
# PREDICTION (before the run): both given networks come back level with the twin's rollouts (1 s 24-30 mm, area
# median ~1.0); the twin with material given within 2 mm of its hidden-material rollouts (it ignores a given material).
# READ: still blown up = the given recipe really does cost rollouts, and the scoring mismatch is not the whole story.
#   $1 = GPU (checked empty twice a minute apart)
cd /home/nas5/jooyeolyun/repos/simulation-assestization
L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
g=$1
u=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g=$g '$1==g{print $2}')
busy() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$u"; }
while busy || { sleep 60; busy; }; do sleep 60; done
echo "[rescore] GPU $g $(date -u)"
for x in "cloth_cov_seven_ownup_given cov_seven_ownup_given_rg" "cloth_cov_seven_ownup_given_rp cov_seven_ownup_given_rp_rg" \
         "cloth_cov_seven_ownup cov_seven_ownup_rg"; do
  set -- $x
  CUDA_VISIBLE_DEVICES=$g $L scripts/l2_coverage_score.py --ckpt outputs/lsim/$1/step11999.pt --tag $2 --roll_given \
    > outputs/lsim/rescore_$2.log 2>&1 &
done
wait
echo "[rescore] scored $(date -u)"
