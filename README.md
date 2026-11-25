# Shoulder of Giants

A learned simulator for the scenes a household robot acts in. Networks predict how explicit 3D meshes move, frame by
frame: rigid objects, hinged parts such as drawers and doors, and cloth. They learn first by copying a classical physics
simulator (NVIDIA Newton 1.7, the *teacher*) across many objects, materials and gripper actions. Then they learn from
real recorded motion, with the aim of ending up more accurate than the teacher. There is no camera in the loop and no
language model in the method. The Python package is `lsim`.

**Start here**
1. The [project brief](https://claude.ai/artifact/5WR9dMDTujSfrNAyhjCEQJ): the main page, a short summary with videos.
   Every other page links from it, and every page's title starts with [SoG].
2. The [test results](https://claude.ai/artifact/FZbzJtpVHJWYpDXgYTGpzj): every chart, table and video, and the runs in
   progress.
3. [Architecture](https://claude.ai/artifact/QDsKHiZi79nEwTZ7wvUh31): the cloth and rigid networks as built, and the
   measurement behind each choice.
4. [Making the training data](https://claude.ai/artifact/Bqgkb28DtrCtLMpWj1hLtt): how the rigid and cloth episodes are
   simulated, recorded and checked.
5. [`docs/DIRECTION.md`](docs/DIRECTION.md): the lab notebook. It holds the design, the plan, and every run with the
   prediction written before it.
6. [`CLAUDE.md`](CLAUDE.md): the working rules and the full current state. The coding agent reads it too.

## Where it stands (2026-11-25)

The one-frame score is the error over one frame divided by the error of simply continuing at constant velocity, on
episodes never trained on. Lower is better, and 1 means no better than that guess.

- **Rigid objects.** A small plain transformer (1.6M parameters, no physics layer, every input given) is the first
  network clearly ahead of the best older baseline. It scores 0.45 over one frame, and its replays of unseen scenes stay
  closer to the teacher. Longer training improves the one-frame fit, and more episodes improve the replays; a bigger
  network does not help.
  - Not yet working: shut drawers drift open, and objects the gripper knocks over stay standing.
  - A real cube does not slide down a real ramp. The ramp data does show objects let go on a slope, but no training
    window starts from rest the way the real test does (`l2_overfit.py --p_start` adds such windows; under test).
- **Cloth.** A 64M transformer predicts a force on each vertex, and an exact physics layer (gravity, F = ma, air drag,
  contact with fixed surfaces, held vertices) moves the cloth. It scores 0.475 over one frame and is 14 / 50 / 77 mm
  from the teacher after 1 / 2 / 3.8 s; 5 of 336 unseen cloths blow up (end above twice their area), against 28 for
  the same training with the material hidden.
  - Ten times the data was worth 0.015 on that score, twice the training 0.068. More than the 20k episodes is not needed
    now.
  - On real cloth (PGND recordings), its shorter-trained version (24k steps) after fine-tuning on real data is about
    18 mm off, against the teacher's 16 mm with fitted material: not yet more real than its teacher. The same real
    fine-tune went unstable on the best network (26 mm), so its settings need adjusting for it.
- **For now networks get every input**: material, gripper force and all eight past frames (rigid
  `l2_overfit.py --plain --aug 1`, cloth `cloth_train --given`). Inferring the hidden ones comes back once the forward
  simulator fits. Cloth trained that way fits one frame as well and running alone stretches a fifth as much (12%
  against 61%). Score a network on the inputs it trained with (`l2_coverage_score.py --roll_given` for `--given`).

## Setup

The code runs on a shared 8× RTX A6000 box (driver CUDA 12.9). Some scripts hard-code this box's paths.

```bash
bash scripts/l0_install_env.sh            # conda env `lsim`: Python 3.12, torch cu128, Newton 1.7, Warp 1.18 +cu12
conda activate lsim && pip install -e . --no-deps
python scripts/l0_teacher_gate.py all     # check the teacher before generating any data
```

Datasets are git-ignored and live in `data/datasets/`:

| dataset | contents | made by |
|---|---|---|
| `v0` | rigid and hinged scenes, 50,048 episodes (391 shards of 128) | `scripts/l1_generate.py` |
| `v0r` | ramp supplement, 12.8k episodes | `scripts/l1_generate.py` |
| `c1`, `c1_test` | cloth, 20,032 + 1,024 episodes | `scripts/l1_generate_cloth.py` |

## The current best networks

```bash
# Rigid (outputs/lsim/rigid_pilot1024p_attn_direct_36k.pt): about 7 h on one GPU
python scripts/l2_overfit.py --data data/datasets/v0 --episodes 1024 --steps 36000 --plain --trunk attn --d 128 \
  --layers 6 --direct --pair_norm --coulomb --pf 6 --noise 0.003,0.015 --mix --out outputs/lsim/rigid.pt
python scripts/l2_small_fit_rigid.py --ckpt outputs/lsim/rigid.pt --episodes 32 --shard 8 --tag held   # unseen shard
# For new rigid runs add --aug 1 (a random turn of each scene): at 128 episodes it closed the gap to unseen episodes.

# Cloth (outputs/lsim/cloth_q64_ownup_cont2_given/step23999.pt): three 24,000-step stretches of this command, each
# from the last one's step23999.pt with --init <path>. The first two ran without --given, the third with it (a run
# with --given from the start is untested at full data). 72,000 steps on two GPUs, about 51 h.
torchrun --nproc_per_node 2 -m lsim.train.cloth_train --data data/datasets/c1 --arch qwen --d 512 --layers 16 \
  --heads 8 --kv_heads 4 --mlp 1536 --steps 24000 --lr 6e-4 --batch 4 --sig 1e-3 --pf 4 --prob_slots --unc --accel \
  --layer 2 --edge 1.0 --side_apart --own_up --given --save_every 4000 --keep --val "data/datasets/c1_test/shard_*.npz" \
  --out outputs/lsim/cloth
python scripts/l2_coverage_score.py --ckpt outputs/lsim/cloth/step23999.pt --tag cloth --roll_given   # c1_test
```

## Layout

| path | contents |
|---|---|
| `src/lsim/` | the package: `teacher/` (Newton wrappers), `data/` (scene and episode generation), `train/` (windows, losses, training loops), `model/` (networks, the exact physics layer, rollouts), `real/` (real recordings) |
| `scripts/` | `l0_*` environment and teacher checks · `l1_*` data generation and checks · `l2_*` training, scoring, rendering |
| `outputs/lsim/*.sh` | run scripts; each has its prediction in its header (runs and logs themselves are git-ignored) |
| `docs/` | `DIRECTION.md`, the HTML pages, `results/` (score files) |
| `data/real/` | small real-data files (tracked) |
| `tests/` | the physics layer against closed forms |
| `archive/theta_track/` | an earlier project's code and results; read-only, and nothing imports it |

## Working on the shared box

- **GPUs.** Use only GPUs with no process on them, checked twice a minute apart. Give each training run its own GPU,
  and pack evaluations onto one. Billing is per GPU-hour whether or not the GPU is busy, so never hold an idle one.
- **Disk.** The local disk (`/tmp`, `~`) is nearly full, so keep large files under `outputs/` on the NAS.
- **CPU worker pools.** Set `CUDA_VISIBLE_DEVICES=""` and use `spawn`. A forked pool once locked the GPU driver for
  every user.
- **Runs.** Write the prediction into the run script before starting it. Compare runs paired, with an interval, never
  on a single checkpoint.
