"""Fingerprints of the existing cloth checkpoints' outputs on a fixed batch, for tests/test_cloth_qwen.py: a change to
the cloth path (windows, collate, network, exact layer) must leave every old checkpoint's outputs bit-for-bit the same.

    CUDA_VISIBLE_DEVICES="" python tests/cloth_ref_outputs.py --write   # once, with the code the fingerprints describe

The batch: the smallest cloth of the first c1_test shard, one training window (seeded collate, augmentation and slot
hiding on), and a 3-frame rollout from frame 40 (the cloth lifted and moving) through lsim.train.cloth_train.rollout (no augmentation). On CPU in fp32 (bf16=False: the CPU has no autocast for the bf16 residual stream)."""
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests/data/cloth_ref_outputs.json"
CKPTS = ["outputs/lsim/cloth_v3/last.pt", "outputs/lsim/cloth_q64/step23999.pt", "outputs/lsim/cloth_rq64/step1000.pt"]


def _h(t):
    a = t.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:24] + f" {a.dtype} {list(a.shape)} max {float(np.abs(a).max()):.6g}"


def episode():
    from lsim.train.cloth_windows import load_cloth_shard
    shard = sorted(glob.glob(str(ROOT / "data/datasets/c1_test/shard_*.npz")))[0][:-4]
    eps = load_cloth_shard(shard)
    return min(eps, key=lambda e: (len(e.rest), e.w))


def fingerprints(ck_path):
    from lsim.model.cloth_net import build_cloth_model, cloth_step, decode
    from lsim.train.cloth_train import contact_kw, forces, kin_next, rollout
    from lsim.train.cloth_windows import collate_cloth, make_cloth_window
    torch.manual_seed(0)
    c = torch.load(ROOT / ck_path, map_location="cpu", weights_only=False)
    m = build_cloth_model(c["args"], ckpt=False, bf16=False)   # CPU: no autocast
    m.load_state_dict(c["model"])
    m.eval()
    ep = episode()
    rng = np.random.default_rng(7)
    win = make_cloth_window(ep, 45, 8, 8, rng)
    b = collate_cloth([win], "cpu", rng)
    Fw = b["pos"].shape[1]
    b.update(tau=torch.zeros(Fw), frame_given=torch.ones(Fw))
    b["kin_next"] = kin_next(b, torch.cat([b["pos"], b["pos"][:, -1:]], 1))
    out = {}
    with torch.no_grad():
        fi, fc, pr, ec = m(b)
        out.update(f_int=_h(fi), f_con=_h(fc), params=_h(pr), ec=_h(ec))
        w, mu = decode(pr, b)
        F = forces(b, fi, fc, w)
        kw = contact_kw(m, b, fc, w)
        kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, 7]
        free = b["is_cloth"] & ~b["held"][:, 7]
        x = cloth_step(b, b["pos"][:, 6], b["pos"][:, 7], F[:, 7], w, ~free, b["pos"][:, 8], mu, **kw)
        out["step"] = _h(x)
        pred, true, carried = rollout(m, ep, 40, 3, dev="cpu")
        out["rollout"] = _h(torch.as_tensor(pred))
    return out


def main():
    res = {ck: fingerprints(ck) for ck in CKPTS if (ROOT / ck).exists()}
    print(json.dumps(res, indent=1))
    if "--write" in sys.argv:
        REF.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
