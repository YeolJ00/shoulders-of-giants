"""WHEN does the clip go wrong? Block masks on the corrected (masked-context) path.

This is the thread worth pursuing, for two reasons the detection results made clear:

  * localisation works where detection fails. Differencing a violation's mask profile
    against its own source recovered the authored moment to within 4 frames on 58% of
    violations (chance ~16%), and it was BEST on teleport -- the violation that detection
    misses entirely. The model sees the discontinuity precisely; the aggregate statistic
    was averaging it away.
  * a localised signal is what assigns blame to a PARAMETER. A peak at impact implicates
    contact damping; a peak during the slide implicates friction. That is the only route
    from a plausibility signal to a per-parameter gradient.

Differencing against the clip's own source is essential, not cosmetic: the mask sweep has
a strong edge artefact (peaks pile up at the first and last mask) which both members share
and which cancels exactly.

Only violation clips and their sources are scored -- 36 of 72 -- since the rest have no
authored moment to check against.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v4_localize.py <input.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

VJ = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_repo"
sys.path.insert(0, VJ)
SIZE, PATCH, TUBELET, FRAMES = 256, 16, 2, 64
GRID = SIZE // PATCH; TPS = GRID * GRID; DEPTH = FRAMES // TUBELET
WIDTH = 2                                   # mask two slots (4 frames) at a time

# where corrupt() in v1_killtest actually puts each violation, in slots
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


def main():
    import torch
    import torch.nn.functional as F
    from src.hub.backbones import _make_vjepa2_model

    src = Path(sys.argv[1]); clips = json.loads(src.read_text())["clips"]
    need = {k for k, r in clips.items() if r.get("probe") == "violation"}
    need |= {clips[k]["pair_with"] for k in need if clips[k].get("pair_with")}
    print(f"{len(need)} clips (violations + their sources)", flush=True)

    enc_c, pred = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_t, _ = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_c, enc_t, pred = (m.to("cuda:0").eval() for m in (enc_c, enc_t, pred))
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)

    def enc(x, model, ids=None):
        with torch.no_grad():
            h = model(x, masks=None if ids is None else [ids])
        return h[0] if isinstance(h, (list, tuple)) else h

    def profile(frames):
        """One L1 per masked window; the target is hidden and predicted from BOTH sides."""
        x = torch.as_tensor(frames, dtype=torch.float32).div_(255.)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        x = ((x - mean) / std).to("cuda:0")
        h = enc(x, enc_t); hn = F.layer_norm(h, (h.size(-1),))
        allids = np.arange(DEPTH * TPS)
        out, centres = [], []
        for t0 in range(0, DEPTH - WIDTH + 1):
            tgt = np.arange(t0 * TPS, (t0 + WIDTH) * TPS)
            ctx = np.setdiff1d(allids, tgt)
            c = torch.as_tensor(ctx, device=h.device).unsqueeze(0)
            t = torch.as_tensor(tgt, device=h.device).unsqueeze(0)
            ct = enc(x, enc_c, ids=c)                       # masked context encode
            with torch.no_grad():
                z = pred(ct, [c], [t])
            z = z[0] if isinstance(z, (list, tuple)) else z
            g = torch.gather(hn, 1, t.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
            out.append(float(F.l1_loss(z.float(), g.float())))
            centres.append(t0 + WIDTH / 2.0)
        return np.asarray(out), np.asarray(centres)

    prof = {}
    for i, k in enumerate(sorted(need)):
        p, centres = profile(read_clip(clips[k]["clip"]))
        prof[k] = p.tolist()
        print(f"  [{i+1}/{len(need)}] {k:<34} peak@slot {centres[int(np.argmax(p))]:.1f}",
              flush=True)

    print("\nDIFFERENTIAL localisation (violation profile minus its own source)")
    errs = []
    for k, r in sorted(clips.items()):
        if r.get("probe") != "violation" or r.get("pair_with") not in prof:
            continue
        truth = TRUTH.get(r["param"])
        a, b = np.asarray(prof[k]), np.asarray(prof[r["pair_with"]])
        pk = float(centres[int(np.argmax(a - b))])
        if truth is not None:
            errs.append(abs(pk - truth))
        print(f"  {r['param']:<10} peak@{pk:5.1f}   authored@"
              f"{'n/a' if truth is None else f'{truth:5.1f}'}"
              f"   {'' if truth is None else f'err {abs(pk-truth):5.1f}'}")
    if errs:
        hit = sum(e <= 2 for e in errs)
        print(f"\nmedian error {np.median(errs):.1f} slots ({2*np.median(errs):.0f} frames)"
              f"   within 2 slots: {hit}/{len(errs)} = {100*hit/len(errs):.0f}%"
              f"   (chance ~16%, uniform guess ~8 slots)")
    dst = Path("docs/results") / f"v4_localize_{src.stem.replace('v21_input_','')}.json"
    dst.write_text(json.dumps({"model": "vjepa2_vit_large", "grid": GRID,
                               "width": WIDTH, "centres": centres.tolist(),
                               "profiles": prof}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
