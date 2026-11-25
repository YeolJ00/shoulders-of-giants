"""Where does the trunk's gradient come from? (2026-11-21)

The cloth position loss is distance / 1 cm + squared distance / 1 cm^2 per free vertex and chunk step. At the
networks' errors (moving cloth ~1 mm, resting cloth ~10-50 um, the noised chunk steps ~0.1-1 mm) the distance term's
pull on a vertex is 1 / cm whatever its error -- 5x the squared term's at 1 mm -- so near-exact vertices pull as hard
as the moving cloth the metric scores, in the direction of their micrometre errors. Hypothesis (gradient noise): that
pull is mostly noise, and dominates the trunk's update (every step is clipped from a gradient norm of 10-40 to 1),
which would make the fit slow in steps, indifferent to width, and level with train = test.

On training windows with the training recipe (chunk noise 1e-3, pushforward up to 4), the trunk's gradient (side
heads excluded) is split into four parts -- the distance and squared terms, each over moving vertices (speed > 2 cm/s
in the step into the frame) and the rest -- plus the edge term; for each part its norm per batch, its cosine with the
moving-vertex parts within a batch, and its consistency (mean cosine between batches: 1 = all signal, 0 = noise).

    CUDA_VISIBLE_DEVICES=<gpu> $L scripts/l2_grad_split.py --ckpt outputs/lsim/cloth_q64_cont/step23999.pt
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
    ap.add_argument("--data", default="c1")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--tag", default="q64_cont")
    a = ap.parse_args()
    from l2_real_cloth_pgnd_eval import load_model
    from lsim.train.cloth_train import _windows, step_loss
    from lsim.train.cloth_windows import collate_cloth, load_shard
    import torch._functorch.config
    torch._functorch.config.donated_buffer = False                       # several backward passes through one graph
    dev = "cuda"
    paths = sorted(glob.glob(str(REPO / f"data/datasets/{a.data}/shard_*.npz")))[:a.shards]
    eps = [e for p in paths for e in load_shard(p[:-4])]
    m = load_model(a.ckpt, dev)
    m.train()
    side = lambda n: n.startswith(("unc.", "param."))
    params = [p for n, p in m.named_parameters() if not side(n) and p.requires_grad]
    P, K = 8, 8
    names = ("dist_moving", "dist_rest", "sq_moving", "sq_rest", "edge")
    G = {k: [] for k in names}
    share = {k: [] for k in ("moving_frac", "clean_moving_frac")}
    err_mm = {"moving": [], "rest": []}
    rng = np.random.default_rng(5)
    for bi in range(a.batches):
        wins = _windows(eps, a.batch, P, K, np.random.default_rng(100 + bi))
        b = collate_cloth(wins, dev, rng)
        parts = {}
        step_loss(m, b, np.random.default_rng(bi), K, dev, sig=1e-3, pf=4, edge_w=1.0, parts=parts)
        err2, fr = parts["err2"], parts["free"]
        pos = b["pos"]
        j = torch.arange(P - 1, P - 1 + K, device=dev)
        spd = (pos[:, j] - pos[:, j - 1]).norm(dim=-1) * 30            # (N, K), clean frames
        mov = (spd > 0.02)[fr]
        k_of = torch.arange(K, device=dev)[None].expand(fr.shape[0], -1)[fr]
        n = err2.numel()
        share["moving_frac"].append(float(mov.float().mean()))
        share["clean_moving_frac"].append(float((mov & (k_of == 0)).float().mean()))
        e = (err2.detach() + 1e-12).sqrt() * 1e3
        err_mm["moving"].append(e[mov].cpu().numpy()); err_mm["rest"].append(e[~mov].cpu().numpy())
        d = (err2 + 1e-12).sqrt()
        terms = {"dist_moving": d[mov].sum() / n / 1e-2, "dist_rest": d[~mov].sum() / n / 1e-2,
                 "sq_moving": err2[mov].sum() / n / 1e-4, "sq_rest": err2[~mov].sum() / n / 1e-4,
                 "edge": parts["l_edge"]}
        for k, t in terms.items():
            g = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            G[k].append(torch.cat([(x if x is not None else torch.zeros_like(p)).flatten().float()
                                   for x, p in zip(g, params)]))
        print(f"  batch {bi}: {n} free vertex-steps, moving {share['moving_frac'][-1]:.2f}", flush=True)
    cos = lambda x, y: float((x @ y) / (x.norm() * y.norm() + 1e-30))
    out = {"ckpt": a.ckpt, "batches": a.batches, "batch": a.batch,
           "moving_frac": float(np.mean(share["moving_frac"])), "clean_moving_frac": float(np.mean(share["clean_moving_frac"])),
           "err_mm": {k: {"median": float(np.median(np.concatenate(v))), "mean": float(np.concatenate(v).mean())}
                      for k, v in err_mm.items()}, "parts": {}}
    B = a.batches
    for k in names:
        norms = [float(g.norm()) for g in G[k]]
        cons = [cos(G[k][i], G[k][j]) for i in range(B) for j in range(i + 1, B)]
        mean = torch.stack(G[k]).mean(0)
        out["parts"][k] = {"norm": float(np.mean(norms)), "consistency": float(np.mean(cons)),
                           "norm_of_mean": float(mean.norm()),
                           "cos_with_dist_moving": float(np.mean([cos(G[k][i], G["dist_moving"][i]) for i in range(B)])),
                           "cos_with_sq_moving": float(np.mean([cos(G[k][i], G["sq_moving"][i]) for i in range(B)]))}
    tot = [sum(G[k][i] for k in names) for i in range(B)]
    mv = [G["dist_moving"][i] + G["sq_moving"][i] for i in range(B)]
    out["total"] = {"norm": float(np.mean([float(t.norm()) for t in tot])),
                    "consistency": float(np.mean([cos(tot[i], tot[j]) for i in range(B) for j in range(i + 1, B)])),
                    "cos_with_moving": float(np.mean([cos(tot[i], mv[i]) for i in range(B)])),
                    "moving_consistency": float(np.mean([cos(mv[i], mv[j]) for i in range(B) for j in range(i + 1, B)]))}
    # the mean moving gradient (the signal the metric wants) seen by each part's batch gradient
    gm = torch.stack(mv).mean(0)
    out["total"]["cos_batch_total_with_mean_moving"] = float(np.mean([cos(t, gm) for t in tot]))
    out["total"]["cos_batch_moving_with_mean_moving"] = float(np.mean([cos(x, gm) for x in mv]))
    print(f"free vertex-steps moving {out['moving_frac']:.2f} (clean first step {out['clean_moving_frac']:.3f}); error "
          f"median moving {out['err_mm']['moving']['median']:.3f} mm, rest {out['err_mm']['rest']['median']:.3f} mm")
    for k, r in out["parts"].items():
        print(f"  {k:12s} norm {r['norm']:8.3f}  consistency {r['consistency']:+.3f}  cos with dist_moving "
              f"{r['cos_with_dist_moving']:+.3f}  with sq_moving {r['cos_with_sq_moving']:+.3f}")
    t = out["total"]
    print(f"  total        norm {t['norm']:8.3f}  consistency {t['consistency']:+.3f}  cos with moving {t['cos_with_moving']:+.3f}"
          f"  moving consistency {t['moving_consistency']:+.3f}  batch total vs mean moving "
          f"{t['cos_batch_total_with_mean_moving']:+.3f} (moving alone {t['cos_batch_moving_with_mean_moving']:+.3f})")
    json.dump(out, open(REPO / f"docs/results/l2_grad_split_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
