"""Surprise under the IntPhys2 protocol, with the context leak fixed. STANDALONE.

Four corrections over everything run earlier today, all taken from Meta's own code rather
than reconstructed:

1. MASKED CONTEXT ENCODING. evals/intuitive_physics/eval.py does
       h       = target_encoder(pieces, full_mask)
       context = encoder(pieces, masks_enc)
   -- two separate forwards. Every earlier run here (and HF's VJEPA2Model, which cannot
   pass masks to its encoder at all) gathered the context out of one full-clip encode, so
   context tokens had already attended to the region being predicted. Measured leak on the
   2.1 checkpoint: max |masked - gathered| = 0.043, against an effect size of ~0.4%.

2. SEPARATE ENCODER WEIGHTS. The checkpoint ships `encoder` and `target_encoder`; context
   uses the former, targets the latter.

3. DENSE CONTEXT SWEEP (`max_context_mode`), quoting their comment: "Predict for all
   context smaller than CONTEXT, in case the physics breaking event happens before
   CONTEXT + 1". We do not know when a clip goes wrong, so every split is tried.

4. SLIDING WINDOW. `clip.unfold(2, frames_per_clip, stride)` scores several overlapping
   windows per video rather than one.

Calibration that reframes the whole exercise: IntPhys2 reports that "most models perform
at chance levels (50%)" on its realistic splits, against near-perfect human performance.
The 98% figure belongs to IntPhys 1, which is coloured shapes on plain backgrounds. Our
scenes are textured objects on a wooden table, i.e. the regime where the published state
of the art is chance.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v3_intphys2.py <input.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

VJ = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_repo"
sys.path.insert(0, VJ)

SIZE, PATCH, TUBELET = 256, 16, 2
GRID = SIZE // PATCH; TPS = GRID * GRID
FPC = 48                    # IntPhys2: frames_per_clip
STRIDE = 8                  # window stride (theirs is 2; ours are 64-frame clips)
CLIP_FRAMES = 64


def read_clip(path, size=SIZE, n=CLIP_FRAMES):
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(str(path)); fr = np.stack([f[..., :3] for f in rd]); rd.close()
    fr = fr[np.linspace(0, len(fr) - 1, n).round().astype(int)]
    t, h, w, _ = fr.shape
    s = min(size / w, size / h); nw, nh = int(round(w * s)), int(round(h * s))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out = np.zeros((t, size, size, 3), np.uint8)
    for i in range(t):
        out[i, oy:oy + nh, ox:ox + nw] = np.asarray(
            Image.fromarray(fr[i]).resize((nw, nh), Image.BILINEAR))
    return out


def main():
    import torch
    import torch.nn.functional as F
    from src.hub.backbones import _make_vjepa2_model

    src = Path(sys.argv[1]); clips = json.loads(src.read_text())["clips"]
    enc_c, pred = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_t, _ = _make_vjepa2_model(model_name="vit_large", img_size=SIZE, pretrained=True)
    enc_c, enc_t, pred = (m.to("cuda:0").eval() for m in (enc_c, enc_t, pred))
    E = enc_c.embed_dim
    depth = FPC // TUBELET
    print(f"V-JEPA 2 ViT-L via Meta's loader: embed {E}, {GRID}x{GRID}, "
          f"{FPC}-frame window, {depth} slots", flush=True)
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)

    def enc(x, model, ids=None):
        with torch.no_grad():
            h = model(x, masks=None if ids is None else [ids])
        return h[0] if isinstance(h, (list, tuple)) else h

    def window_losses(win):
        """Dense context sweep on one 48-frame window; returns one L1 per split."""
        x = torch.as_tensor(win, dtype=torch.float32).div_(255.)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        x = ((x - mean) / std).to("cuda:0")
        h = enc(x, enc_t)
        hn = F.layer_norm(h, (h.size(-1),))
        out = []
        for n in range(1, depth):                       # every split, max_context_mode
            c = torch.arange(0, n * TPS, device=h.device).unsqueeze(0)
            t = torch.arange(n * TPS, depth * TPS, device=h.device).unsqueeze(0)
            ct = enc(x, enc_c, ids=c)                   # masked context encode
            with torch.no_grad():
                z = pred(ct, [c], [t])
            z = z[0] if isinstance(z, (list, tuple)) else z
            g = torch.gather(hn, 1, t.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
            out.append(float(F.l1_loss(z.float(), g.float())))
        return np.asarray(out)

    rows = {}
    for i, (key, r) in enumerate(sorted(clips.items())):
        fr = read_clip(r["clip"])
        best, allv = -1.0, []
        for s in range(0, CLIP_FRAMES - FPC + 1, STRIDE):   # sliding window
            L = window_losses(fr[s:s + FPC])
            allv.append(L); best = max(best, float(L.max()))
        A = np.concatenate(allv)
        rows[key] = {k: r[k] for k in ("probe", "param", "value", "ref") if k in r}
        rows[key].update(pair_with=r.get("pair_with"), pair_kind=r.get("pair_kind"),
                         max=best, mean=float(A.mean()), n_evals=int(A.size))
        print(f"  {key:<34} max {best:.5f}  mean {A.mean():.5f}  ({A.size} evals)", flush=True)
    dst = Path("docs/results") / f"v3_intphys2_{src.stem.replace('v21_input_', '')}.json"
    dst.write_text(json.dumps({"model": "vjepa2_vit_large (Meta loader, masked context)",
                               "protocol": "IntPhys2", "rows": rows}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
