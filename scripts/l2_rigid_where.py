"""Where is a rigid model's one-frame error? (2026-11-23, the 128-episode pilots: every layer model, and the fully
trained v1, sits where THE EXACT LAYER ALONE does -- 1.74 of constant velocity on held-out moving bodies -- because the
recorded gripper force is integrated with nothing to balance it; DIRECTION.md "The rigid layer model starts behind
constant velocity".) Every 4th frame from 8 (l2_small_fit_rigid.py's frames), every dynamic rigid body of the first N
episodes of a shard: the model, the exact layer alone (force heads zeroed; layer models only), constant velocity, and
what the body touches at that frame (contact edges within 3 cm to static surfaces / other dynamic bodies / the
gripper; vertices with a given force).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_rigid_where.py score --ckpt outputs/lsim/train_v1/last.pt --shard 1 --tag v1_held
    $L scripts/l2_rigid_where.py join c_attn_held v1_held b1_held b2_held     (same episodes; outputs/lsim/rigid_where/<tag>.npz)

Run `score` on any change to the layer or to what the network may push on, with a checkpoint of that configuration:
its "layer alone" column is where learning starts from.
"""
import argparse
import copy
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/lsim/rigid_where"


def score(a):
    import torch
    from lsim.data.episode import rotate
    from lsim.model.net import build_model, contact_edges
    from lsim.model.rollout import rollout
    from lsim.train.train import collate
    from lsim.train.windows import PARAMS, load_shard, make_window

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{a.tag}.npz"
    dev = torch.device("cuda")
    ck = torch.load(a.ckpt, map_location=dev)
    model = build_model(ck["args"], len(PARAMS)).to(dev); model.load_state_dict(ck["model"]); model.eval()
    layer = hasattr(model, "force") and not getattr(model, "direct", False)
    bare = None
    if layer:
        bare = copy.deepcopy(model)
        with torch.no_grad():
            for head in [bare.force] + ([bare.vforce] if hasattr(bare, "vforce") else []):
                last = [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1]
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()
    shard = sorted(glob.glob(f"{a.data}/shard_*.npz"))[a.shard][:-4]
    eps = [e for e in load_shard(shard) if e.bodies and any(b.role == "dynamic" for b in e.bodies)][:a.episodes]
    P = 8
    rows = []
    rng = np.random.default_rng(0)
    for k, e in enumerate(eps):
        ids = [b.body for b in e.bodies]
        sel = [i for i, b in enumerate(e.bodies) if b.role == "dynamic" and b.type == "rigid"]
        V = lambda q, i: rotate(q[3:], e.bodies[i].verts.astype(float)) + q[:3]
        T = e.q.shape[0]
        for t0 in range(8, T - 1, 4):
            with torch.no_grad():
                pm, true, _ = rollout(model, e, t0, 1)
                pb = rollout(bare, e, t0, 1)[0] if layer else None
            w = make_window(e, t0, P, 0, rng, p_hide_grip=0.0)
            b = collate([w], dev, rng, p_hist_noise=0.0, augment=False, all_past=True, hidden=())
            Fw = b["pos"].shape[1]
            src, dst = contact_edges(b["pos"], b["body_of"], b["sample_of"], b["role"] > 0)
            now = (src % Fw == P - 1)
            vs, vd = src[now] // Fw, dst[now] // Fw
            bo, role = b["body_of"], b["role"]
            fx = (b["f_ext"][:, P - 1].norm(dim=-1) > 0) & (b["f_state"][:, P - 1] == 1)
            for i in sel:
                vt, vn, vp = V(true[0][i], i), V(e.q[t0, ids[i]], i), V(e.q[t0 - 1, ids[i]], i)
                d = lambda v: 1e3 * float(np.linalg.norm(v - vt, axis=1).mean())
                r = role[vd[bo[vs] == i]]
                rows.append((k, t0, i, d(vn), d(V(pm[0][i], i)), d(V(pb[0][i], i)) if layer else np.nan, d(2 * vn - vp),
                             int((r == 0).sum()), int((r == 2).sum()), int((r == 1).sum()), int(fx[bo == i].sum()),
                             1e3 * float(np.linalg.norm(vn - vp, axis=1).mean())))
        R = np.array(rows); m = (R[:, 0] == k) & (R[:, 3] > 0.1)
        print(f"episode {k}: {int(m.sum())} moving body-frames, network / cv {R[m, 4].sum() / max(R[m, 6].sum(), 1e-9):.3f}", flush=True)
    R = np.array(rows)
    np.savez(out, R=R, cols=np.array("ep frame body step model layer cv e_static e_object e_grip given prev_step".split()))
    mov = R[:, 3] > 0.1
    print(f"all: network / cv {R[mov, 4].sum() / R[mov, 6].sum():.3f}" + (f"   layer alone / cv {R[mov, 5].sum() / R[mov, 6].sum():.3f}" if layer else ""))


def join(a):
    D = {t: np.load(OUT / f"{t}.npz")["R"] for t in a.tags}
    tags = list(D)
    R0 = D[tags[0]]
    for t in tags:
        assert D[t].shape == R0.shape and np.allclose(D[t][:, :4], R0[:, :4]), t
    ep, step, cv = R0[:, 0].astype(int), R0[:, 3], R0[:, 6]
    stat, obj, grip, given, prev = R0[:, 7] > 0, R0[:, 8] > 0, R0[:, 9] > 0, R0[:, 10] > 0, R0[:, 11]
    mov = step > 0.1
    E = {t: D[t][:, 4] for t in tags}
    lay = next((D[t][:, 5] for t in tags if not np.isnan(D[t][:, 5]).all()), None)
    if lay is not None:
        E = {"layer alone": lay, **E}
    names = list(E)
    print(f"{len(R0)} body-frames, {mov.sum()} moving (> 0.1 mm a frame), {len(set(ep))} episodes")
    print("pooled error / constant velocity:  " + "   ".join(f"{n} {E[n][mov].sum() / cv[mov].sum():.3f}" for n in names))
    print("with every body-frame's error capped at constant velocity's:  " + "   ".join(f"{n} {np.minimum(E[n], cv)[mov].sum() / cv[mov].sum():.3f}" for n in names))
    print("share of moving body-frames better than constant velocity:  " + "   ".join(f"{n} {(E[n][mov] < cv[mov]).mean():.2f}" for n in names))
    def table(title, cats):
        print(f"\n{title}\n{'':46s}{'n':>5s} {'step':>7s} {'cv':>7s} " + " ".join(f"{n[:12]:>12s}" for n in names) + "    (mean mm; in brackets the share of that model's total error)")
        for name, m in cats:
            m = m & mov
            if m.sum():
                print(f"{name:46s}{m.sum():5d} {step[m].mean():7.2f} {cv[m].mean():7.2f} " + " ".join(f"{E[n][m].mean():6.2f} ({E[n][m].sum() / E[n][mov].sum():.2f})" for n in names))
    table("by what the body touches (contact edges within 3 cm at the current frame)", (
        ("nothing (flight, or hit from beyond 3 cm)", ~stat & ~obj & ~grip),
        ("the gripper only", ~stat & ~obj & grip),
        ("table / floor only", stat & ~obj & ~grip),
        ("table / floor + gripper", stat & ~obj & grip),
        ("another object (no gripper)", obj & ~grip),
        ("another object + gripper", obj & grip),
        ("-- a given force acts (gripper force / poke)", given),
        ("-- no given force", ~given)))
    acc = np.abs(step - prev)
    table("by the body's true step (mm a frame)", (("0.1 - 1", step < 1), ("1 - 5", (step >= 1) & (step < 5)), ("5 - 20", (step >= 5) & (step < 20)), ("> 20", step >= 20)))
    table("by the change of its step from the frame before (constant velocity's own error scale, mm)", (
        ("< 0.3", cv < 0.3), ("0.3 - 1", (cv >= 0.3) & (cv < 1)), ("1 - 3", (cv >= 1) & (cv < 3)), ("3 - 10", (cv >= 3) & (cv < 10)), ("> 10", cv >= 10)))
    table("was it moving the frame before?", (("at rest before (< 0.1 mm), moving now", prev < 0.1), ("moving before and now", prev >= 0.1)))
    print("\nbody-frames at rest (true step < 0.1 mm): mean error mm  " + "   ".join(f"{n} {E[n][~mov].mean():.2f}" for n in names) + f"   cv {cv[~mov].mean():.2f}   n {int((~mov).sum())}")
    print("\nper episode, error / constant velocity (moving body-frames):")
    print("  ep    n   cv mm  " + " ".join(f"{n[:12]:>12s}" for n in names))
    for k in sorted(set(ep)):
        m = mov & (ep == k)
        if m.sum():
            print(f"  {k:2d} {m.sum():4d} {cv[m].sum():7.1f}  " + " ".join(f"{E[n][m].sum() / max(cv[m].sum(), 1e-9):12.2f}" for n in names))
    for n in names:
        tot = np.array([E[n][mov & (ep == k)].sum() for k in sorted(set(ep))])
        o = np.argsort(-tot)
        print(f"{n}: the 3 / 8 episodes with the most error carry {tot[o[:3]].sum() / tot.sum():.2f} / {tot[o[:8]].sum() / tot.sum():.2f} of it")
    n0 = names[1] if lay is not None else names[0]
    print(f"\nthe 20 body-frames with the largest error of {n0}:  ep frame body  step  prev  cv  | " + "  ".join(names) + "  | edges static/object/gripper  given")
    for j in np.argsort(-np.where(mov, E[n0], 0))[:20]:
        print(f"   {ep[j]:3d} {int(R0[j, 1]):4d} {int(R0[j, 2]):2d}  {step[j]:7.2f} {prev[j]:7.2f} {cv[j]:7.2f} | " + " ".join(f"{E[n][j]:7.2f}" for n in names) + f" | {int(R0[j, 7]):4d} {int(R0[j, 8]):4d} {int(R0[j, 9]):4d}  {int(R0[j, 10])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--data", default="data/datasets/v0")
    s.add_argument("--shard", type=int, default=1, help="l2_overfit.py trains on shard 0: another one is held out")
    s.add_argument("--episodes", type=int, default=32)
    s.add_argument("--tag", required=True)
    j = sub.add_parser("join")
    j.add_argument("tags", nargs="+")
    a = ap.parse_args()
    {"score": score, "join": join}[a.cmd](a)
