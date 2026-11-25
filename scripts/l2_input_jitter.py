"""Does the network use its inputs' sub-millimetre detail? (2026-11-21)

The networks score ~0.63 of constant velocity's one-frame error on moving free cloth, what the teacher scores when it
reads its 3 input frames jittered by ~350 um per vertex (l1_teacher_precision: 0 / 30 / 100 / 300 um -> 0.25 / 0.26 /
0.30 / 0.52). The shortfall is the same everywhere -- normal and in-plane, border and interior, every family, phase
and material within ~0.1 (l2_error_normal, l2_error_by_kind) -- as if the learned map read its inputs coarsely. The
same jitter on the network's inputs: every cloth vertex of the 8 given frames moved by iid Gaussian sigma per axis
(the target and constant velocity's reference stay clean), first-step error / constant velocity on moving free
cloth, and the prediction's change from the unjittered one (as a share of constant velocity's error). If its error
stays ~flat where the teacher's rises, the network does not use the detail the teacher needs; if it rises as fast, it
reads the detail and maps it wrongly. l2_error_where's 300 c1_test windows (rng 3), every slot given.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_input_jitter.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt --tag q64_cont
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="c1_test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sigmas_um", default="0,10,30,100,300,1000")
    ap.add_argument("--frames", type=int, default=8, help="jitter the last this many given frames")
    ap.add_argument("--mix", default="all", choices=("all", "fine_clean", "strains_clean", "only_fine"),
                    help="which neighbour features see the jitter: all; fine_clean = all but asinh(1e5 strain) (kept at "
                         "its unjittered value); strains_clean = all but both strain features; only_fine = only "
                         "asinh(1e5 strain) (every other input clean)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    from l2_error_where import classify
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard
    dev = "cuda"
    eps = [e for p in sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz"))) for e in load_shard(p[:-4])]
    wins = _windows(eps, a.n, 8, 8, np.random.default_rng(3))
    m = load_model(a.ckpt, dev)
    P = 8
    from lsim.model.cloth_qwen import N_SLOT, SLOT_F
    state = {"clean": True, "f": None}
    orig = m._slots

    def slots(batch, pos, vel):                                          # neighbour features, mixed per --mix
        f = orig(batch, pos, vel)
        if a.mix == "all":
            return f
        if state["clean"]:
            state["f"] = f
            return f
        sh = f.shape
        fj, fc = f.view(sh[0], sh[1], N_SLOT, SLOT_F), state["f"].view(sh[0], sh[1], N_SLOT, SLOT_F)
        if a.mix == "fine_clean":
            out = fj.clone(); out[..., 7] = fc[..., 7]
        elif a.mix == "strains_clean":
            out = fj.clone(); out[..., 6:8] = fc[..., 6:8]
        else:                                                            # only_fine: f is from the clean frames
            fj = orig(batch, batch["pos_jit"], vel).view(sh[0], sh[1], N_SLOT, SLOT_F)
            out = fc.clone(); out[..., 7] = fj[..., 7]
        return out.reshape(sh)
    m._slots = slots
    sig = [float(s) * 1e-6 for s in a.sigmas_um.split(",")]
    E = {s: 0.0 for s in sig}; D = {s: 0.0 for s in sig}; C = 0.0
    with torch.no_grad():
        for i, w in enumerate(wins):
            b0 = collate_cloth([w], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
            pos = b0["pos"].double()
            tgt, cv = pos[:, P], 2 * pos[:, P - 1] - pos[:, P - 2]
            moving = (pos[:, P - 1] - pos[:, P - 2]).norm(dim=-1) * 30 > 0.02
            cls = classify(b0, P)["free"]
            g = torch.Generator(device=dev).manual_seed(1000 + i)
            x0, sel = None, None
            for s in sig:
                state["clean"] = s == 0
                b = dict(b0)
                if s > 0:
                    p = b0["pos"].clone()
                    cl = b0["is_cloth"]
                    f0 = P - a.frames
                    p[cl, f0:P] += s * torch.randn(p[cl, f0:P].shape, device=dev, generator=g)
                    b["pos"] = p
                    if a.mix == "only_fine":                             # every other input from the clean frames
                        b["pos_jit"] = p
                        b["pos"] = b0["pos"]
                parts = {}
                step_loss(m, b, np.random.default_rng(i), 8, dev, sig=0.0, parts=parts)
                x = parts["x"][:, 0].double()
                if sel is None:
                    sel = parts["free"][:, 0] & moving & cls
                    if not sel.any():
                        break
                    C += float((cv - tgt)[sel].norm(dim=-1).sum())
                    x0 = x
                E[s] += float((x - tgt)[sel].norm(dim=-1).sum())
                D[s] += float((x - x0)[sel].norm(dim=-1).sum())
            if i % 100 == 0:
                print(f"  {i} windows", flush=True)
    teacher = {0: 0.248, 30: 0.26, 100: 0.30, 300: 0.52, 1000: 3.8}
    out = {"ckpt": a.ckpt, "frames": a.frames, "rows": []}
    for s in sig:
        r = {"sigma_um": 1e6 * s, "ratio": E[s] / C, "change": D[s] / C}
        out["rows"].append(r)
        t = teacher.get(round(1e6 * s))
        print(f"  {1e6 * s:6g} um: error / constant velocity {r['ratio']:.3f}, prediction moved {r['change']:.3f} of constant "
              f"velocity's error" + (f"   (teacher from 3 frames {t})" if t is not None else ""))
    json.dump(out, open(REPO / f"docs/results/l2_input_jitter_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
