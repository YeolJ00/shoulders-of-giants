"""8 frames per call against 1 frame per call, on cloth (2026-11-06).

From the teacher's true 8-frame history, each checkpoint produces the next 8 frames in its own mode -- a --chunk model
in one call (cloth_train.chunk_inputs / chunk_steps), a per-frame model in 8 calls, each fed its own previous
predictions (as cloth_train.rollout) -- and is scored at each coming frame k = 1..8:
  pos_mm    mean free-vertex distance to the teacher (mm)
  shape_pct mean over mesh edges with a free end of |predicted length - teacher's length| / rest length (%)
on 128 windows from TRAINING shards (the dataset's first 4) and 128 from held-out shards, so it doubles as a train /
held-out curve. Also the teacher's own per-frame change of edge length, for scale.

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_chunk_fit.py --runs cloth_ck8,cloth_ck1 --train c1 --test c1_test
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
P = K = 8


@torch.no_grad()
def predict8(model, b, dev):
    """(N, K, 3): the 8 coming frames, in the model's own mode."""
    from lsim.model.cloth_net import cloth_step, decode, kin_next
    from lsim.train.cloth_train import DT, chunk_inputs, chunk_steps, contact_kw, forces
    pos = b["pos"]
    free = b["is_cloth"][:, None] & ~b["held"]
    if getattr(model, "chunk", 0):
        kn = kin_next(b, torch.cat([pos, pos[:, -1:]], 1))
        fi, fc, pr, _ = model(chunk_inputs(b, pos, free, kn, P, K, dev))
        w, mu = decode(pr, b)
        return chunk_steps(b, pos, forces(b, fi, fc, w), contact_kw(model, b, fc, w), w, free, mu, P, K)
    x = pos.clone()
    carried = None
    for k in range(K):                                                   # as cloth_train.rollout, on the window
        n = P + k
        lo = n - P
        xw = x[:, lo:n]
        pv = x[:, lo - 1:n - 1] if lo > 0 else torch.cat([xw[:, :1] - b["vel"][:, :1] * DT, xw[:, :-1]], 1)
        bj = dict(b, pos=xw, vel=(xw - pv) / DT, absent=b["absent"][:, lo:n], f_ext=b["f_ext"][:, lo:n],
                  f_state=b["f_state"][:, lo:n], held=b["held"][:, lo:n], tau=torch.zeros(P, device=dev),
                  frame_given=torch.ones(P, device=dev), n_past=P)
        bj["kin_next"] = kin_next(bj, pos[:, lo:n + 1])
        fi, fc, pr, _ = model(bj)
        carried = pr if carried is None else carried
        w, mu = decode(carried, b)
        kw = contact_kw(model, bj, fc, w)
        kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, -1]
        x[:, n] = cloth_step(b, x[:, n - 2], x[:, n - 1], forces(bj, fi, fc, w)[:, -1], w, ~free[:, n - 1], pos[:, n], mu, **kw)
    return x[:, P:P + K]


@torch.no_grad()
def score(model, wins, dev):
    from lsim.train.cloth_windows import collate_cloth
    pos_k, shape_k, teach = [], [], []
    for i in range(0, len(wins), 4):
        b = collate_cloth(wins[i:i + 4], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
        x = predict8(model, b, dev)
        tgt = b["pos"][:, P:P + K]
        free = (b["is_cloth"][:, None] & ~b["held"])[:, P - 1:P - 1 + K]
        d = (x - tgt).norm(dim=-1)
        pos_k.append([1e3 * d[:, k][free[:, k]].mean().item() for k in range(K)])
        E = b["edges"]
        ei, ej = E[:, 0], E[:, 1]
        L0 = (b["rest"][ei] - b["rest"][ej]).norm(dim=-1) * b["size"][b["sample_of"][ei]]
        dl = ((x[ei] - x[ej]).norm(dim=-1) - (tgt[ei] - tgt[ej]).norm(dim=-1)).abs() / L0[:, None]
        fe = free[ei] | free[ej]
        shape_k.append([100 * dl[:, k][fe[:, k]].mean().item() for k in range(K)])
        lt = (b["pos"][ei, P - 1:P + K] - b["pos"][ej, P - 1:P + K]).norm(dim=-1)
        teach.append(100 * ((lt[:, 1:] - lt[:, :-1]).abs() / L0[:, None]).mean().item())
    return np.mean(pos_k, 0).tolist(), np.mean(shape_k, 0).tolist(), float(np.mean(teach))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="outputs/lsim/<run>/step*.pt, comma-separated")
    ap.add_argument("--train", default="c1")
    ap.add_argument("--test", default="c1_test")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="l2_chunk_fit")
    a = ap.parse_args()
    from l2_fit_curve import windows
    from l2_real_cloth_pgnd_eval import load_model
    dev = "cuda"
    sets = {"train": windows(a.train, a.n, 1), "test": windows(a.test, a.n, 2)}
    out = []
    for run in a.runs.split(","):
        ck = sorted(glob.glob(str(REPO / f"outputs/lsim/{run}/step*.pt")), key=lambda p: int(re.findall(r"step(\d+)", p)[-1]))
        for p in ck:
            step = int(re.findall(r"step(\d+)", p)[-1])
            m = load_model(p, dev)
            row = {"run": run, "step": step, "chunk": int(getattr(m, "chunk", 0))}
            for k, w in sets.items():
                pk, sk, t = score(m, w, dev)
                row[k] = {"pos_mm": pk, "shape_pct": sk, "teacher_edge_change_pct": t}
                print(f"{run} step {step:5d} {k:5s} pos mm  " + " ".join(f"{v:6.2f}" for v in pk) +
                      "   shape %  " + " ".join(f"{v:5.2f}" for v in sk) + f"   (teacher {t:.4f} %)", flush=True)
            out.append(row)
            del m; torch.cuda.empty_cache()
    json.dump(out, open(REPO / f"docs/results/{a.out}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
