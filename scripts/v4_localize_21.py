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
import os as _os
# ViT-g (1B, 1408 wide) by default; ViT-G (2B, 1664 wide) is the TEACHER the distilled
# ViT-B was trained against -- its width is exactly the pred_teacher_embed_dim: 1664 that
# made the distilled checkpoint unusable.
_ARCH = _os.environ.get("V21_ARCH", "giant")          # "giant" | "gigantic"
ARCH = f"vjepa2_1_vit_{_ARCH}_384"
CKPT = ("/home/nas5/jooyeolyun/repos/simulation-assestization/models/"
        + ("vjepa2_1_vitG_384.pt" if _ARCH == "gigantic" else "vjepa2_1_vitg_384.pt"))
sys.path.insert(0, VJ)
# 384 / patch 16 -> 24x24 = 576 spatial patches, 2.25x the density of the 256 models.
# Everything else -- block masks, masked context encode, differencing against the clip's
# own source -- is identical to v4_localize.py, so any difference in localisation accuracy
# is attributable to feature density and not to the readout.
SIZE, PATCH, TUBELET, FRAMES, LEVELS = 384, 16, 2, 64, 4
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
    from src.hub.backbones import _make_vjepa2_1_model

    src = Path(sys.argv[1]); clips = json.loads(src.read_text())["clips"]
    need = {k for k, r in clips.items() if r.get("probe") == "violation"}
    need |= {clips[k]["pair_with"] for k in need if clips[k].get("pair_with")}
    print(f"{len(need)} clips (violations + their sources)", flush=True)

    mk = lambda: _make_vjepa2_1_model(model_name=ARCH, img_size=SIZE,
                                      predictor_num_mask_tokens=8,
                                      n_output_distillation=LEVELS,
                                      return_all_tokens=True, pretrained=False)
    enc_c, pred = mk(); enc_t, _ = mk()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    def strip(sd, *p):
        o = {}
        for k, v in sd.items():
            for q in p:
                if k.startswith(q):
                    k = k[len(q):]
            o[k] = v
        return o
    for nm, mod, key in (("context", enc_c, "encoder"), ("target", enc_t, "target_encoder"),
                         ("predictor", pred, "predictor")):
        r = mod.load_state_dict(strip(ck[key], "module.", "backbone."), strict=False)
        if r.missing_keys:
            raise RuntimeError(f"{nm}: {len(r.missing_keys)} missing")
    del ck
    enc_c, enc_t, pred = (m.to("cuda:0").eval() for m in (enc_c, enc_t, pred))
    E = enc_c.embed_dim
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)

    def enc(x, model, ids=None):
        with torch.no_grad():
            h = model(x, masks=None if ids is None else [ids], training=True)
        return h[0] if isinstance(h, (list, tuple)) else h

    def profile(frames):
        """One L1 per masked window; the target is hidden and predicted from BOTH sides."""
        x = torch.as_tensor(frames, dtype=torch.float32).div_(255.)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        x = ((x - mean) / std).to("cuda:0")
        h = enc(x, enc_t)
        # 2.1 recipe: the 4-level target splits into chunks, each normed SEPARATELY
        hn = torch.cat([F.layer_norm(h[:, :, i*E:(i+1)*E], (E,)) for i in range(LEVELS)], 2)
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
    dst = Path("docs/results") / f"v4loc21_{src.stem.replace('v21_input_','')}.json"
    dst.write_text(json.dumps({"model": ARCH, "grid": GRID,
                               "width": WIDTH, "centres": centres.tolist(),
                               "profiles": prof}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
