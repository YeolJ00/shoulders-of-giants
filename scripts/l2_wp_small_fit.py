"""WorldParticle's small-fit test: one-frame error over constant velocity on the episodes it trained on (every frame,
free cloth vertices that move more than 0.1 mm in the frame). Run in the WorldParticle env:

    ATTN_IMPL=flash_attn CUDA_VISIBLE_DEVICES=<gpu> /home/nas5/jooyeolyun/envs/worldparticle/bin/python \
        scripts/l2_wp_small_fit.py --ckpt outputs/lsim/wp/<run>/last.pt --data outputs/lsim/wp/data/<dir>

Also prints how much the learned correction varies across the cloth against how much it should (a network that
ignores its input gives one constant correction: the 2026-11-24 full run cancelled its own gravity step and nothing
else).
"""
import argparse
import glob
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from l2_wp_train import DT, Episode, build, step  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(REPO / a.ckpt, map_location=dev)
    model = build(ck["cfg"], dev)
    model.model.load_state_dict(ck["model"])
    model.eval()
    g = torch.tensor([0.0, 0.0, -9.81], device=dev)
    em = ec = 0.0
    spread_got, spread_need = [], []
    for f in sorted(glob.glob(str(REPO / a.data / "*.npz"))):
        e = Episode(f, dev)
        for t in range(1, e.T - 1):
            pos, vel = e.state(t)
            _, _, p, _ = step(model, e, pos, vel, t)
            tgt = e.x[t + 1]
            cv = 2 * e.x[t] - e.x[t - 1]
            sel = ~e.driven(t)
            sel[e.Vc:] = False
            sel &= (tgt - e.x[t]).norm(dim=-1) > 1e-4
            if sel.sum() < 10:
                continue
            em += float((p - tgt).norm(dim=-1)[sel].sum())
            ec += float((cv - tgt).norm(dim=-1)[sel].sum())
            base = pos + DT * vel + 0.5 * DT * DT * g                      # its integrator alone
            spread_got.append(float((p - base)[sel].std(0).mean()))
            spread_need.append(float((tgt - base)[sel].std(0).mean()))
    print(f"{a.ckpt}: one frame {em / max(ec, 1e-12):.3f} of constant velocity on its own frames; correction spread "
          f"across the cloth {1e3 * sum(spread_got) / len(spread_got):.3f} mm (needed "
          f"{1e3 * sum(spread_need) / len(spread_need):.3f})", flush=True)


if __name__ == "__main__":
    main()
