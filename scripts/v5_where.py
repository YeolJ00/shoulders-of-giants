"""WHERE, not just when: spatio-temporal masks.

`when` is settled -- block masks over time, differenced against the clip's own source,
recover the authored moment to 1 slot median (62% within 4 frames, chance ~16%), and are
exact on teleport, the violation detection misses entirely.

`where` is the untested half, and it is the one that assigns blame. A peak that lands ON
THE OBJECT at the impact frame implicates contact damping; one that lands on the object
during the slide implicates friction; one that lands on the background implicates nothing
and tells us the signal is reading scenery.

Ground truth exists for both axes and neither is chosen after the fact:
  WHEN  -- the frame corrupt() authored the violation at
  WHERE -- the object's projected patches, from the same camera the crop uses

Masks are a spatial block x a temporal window; the context is everything else, re-encoded
with the mask (the leak fix). Differencing against the source cancels the sweep's edge
artefact, which both members share.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v5_where.py <input.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

VJ = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_repo"
sys.path.insert(0, VJ)
SIZE, PATCH, TUBELET, FRAMES = 256, 16, 2, 64
GRID = SIZE // PATCH; TPS = GRID * GRID; DEPTH = FRAMES // TUBELET
BLK, BSTRIDE = 4, 4          # 4x4-patch spatial block, strided by 4 -> 16 positions
TW, TSTRIDE = 2, 2           # 2-slot temporal window -> 15 positions
TRUTH = {"teleport": 16.0, "vanish": 18.0, "freeze": 10.5, "through": 10.5, "rewind": None}


def read_clip(path):
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(str(path)); fr = np.stack([f[..., :3] for f in rd]); rd.close()
    fr = fr[np.linspace(0, len(fr) - 1, FRAMES).round().astype(int)]
    t, h, w, _ = fr.shape
    s = min(SIZE / w, SIZE / h); nw, nh = int(round(w * s)), int(round(h * s))
    ox, oy = (SIZE - nw) // 2, (SIZE - nh) // 2
    out = np.zeros((t, SIZE, SIZE, 3), np.uint8)
    for i in range(t):
        out[i, oy:oy + nh, ox:ox + nw] = np.asarray(
            Image.fromarray(fr[i]).resize((nw, nh), Image.BILINEAR))
    return out


def blocks():
    """(name, spatial_patch_ids, row, col) for each spatial block."""
    out = []
    for r in range(0, GRID - BLK + 1, BSTRIDE):
        for c in range(0, GRID - BLK + 1, BSTRIDE):
            ids = np.array([(r + i) * GRID + (c + j) for i in range(BLK) for j in range(BLK)])
            out.append((f"r{r}c{c}", ids, r + BLK / 2.0, c + BLK / 2.0))
    return out


def main():
    import torch
    import torch.nn.functional as F
    from src.hub.backbones import _make_vjepa2_model

    src = Path(sys.argv[1]); clips = json.loads(src.read_text())["clips"]
    need = {k for k, r in clips.items() if r.get("probe") == "violation"}
    need |= {clips[k]["pair_with"] for k in need if clips[k].get("pair_with")}
    B = blocks()
    T = list(range(0, DEPTH - TW + 1, TSTRIDE))
    print(f"{len(need)} clips x {len(B)} spatial blocks x {len(T)} time windows "
          f"= {len(B)*len(T)} masks each", flush=True)

    enc_c, pred = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_t, _ = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_c, enc_t, pred = (m.to("cuda:0").eval() for m in (enc_c, enc_t, pred))
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
    allids = np.arange(DEPTH * TPS)

    def enc(x, model, ids=None):
        with torch.no_grad():
            h = model(x, masks=None if ids is None else [ids])
        return h[0] if isinstance(h, (list, tuple)) else h

    def grid_profile(frames):
        x = torch.as_tensor(frames, dtype=torch.float32).div_(255.)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        x = ((x - mean) / std).to("cuda:0")
        h = enc(x, enc_t); hn = F.layer_norm(h, (h.size(-1),))
        M = np.zeros((len(T), len(B)), np.float32)
        for ti, t0 in enumerate(T):
            for bi, (_, sp, _, _) in enumerate(B):
                tgt = np.concatenate([s * TPS + sp for s in range(t0, t0 + TW)])
                ctx = np.setdiff1d(allids, tgt)
                c = torch.as_tensor(ctx, device=h.device).unsqueeze(0)
                t = torch.as_tensor(tgt, device=h.device).unsqueeze(0)
                ct = enc(x, enc_c, ids=c)
                with torch.no_grad():
                    z = pred(ct, [c], [t])
                z = z[0] if isinstance(z, (list, tuple)) else z
                g = torch.gather(hn, 1, t.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
                M[ti, bi] = float(F.l1_loss(z.float(), g.float()))
        return M

    prof = {}
    for i, k in enumerate(sorted(need)):
        prof[k] = grid_profile(read_clip(clips[k]["clip"])).tolist()
        print(f"  [{i+1}/{len(need)}] {k}", flush=True)

    dst = Path("docs/results") / f"v5_where_{src.stem.replace('v21_input_','')}.json"
    dst.write_text(json.dumps({"model": "vjepa2_vit_large", "grid": GRID, "blk": BLK,
                               "t_windows": T, "tw": TW,
                               "blocks": [[b[0], float(b[2]), float(b[3])] for b in B],
                               "profiles": prof}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
