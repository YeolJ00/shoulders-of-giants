"""Score clips with V-JEPA 2.1 ViT-g (non-distilled). STANDALONE by necessity.

Meta's vjepa2 repo ships its own top-level `src` package, which shadows ours, so this
script imports nothing from the project -- it takes everything precomputed from
scripts/v21_prep.py and inlines its own clip loading.

Why this checkpoint: the HuggingFace port is `..._dist_vitG_...`, a ViT-B DISTILLED from
ViT-G, whose predictor emits into the teacher's 1664-d space and so cannot be compared
against its own 768-d encoder. ViT-g carries no `dist` tag because it IS a teacher --
verified by loading: encoder 1408 wide, predictor out 5632 = 4 x 1408, four
Deep-Self-Supervision levels at the encoder's own width.

The target recipe is copied from app/vjepa_2_1/train.py::forward_target rather than
inferred, because two details would have been wrong if guessed:
  * the 5632-wide hierarchical feature splits into four 1408 chunks, each layer-normed
    SEPARATELY (not the whole vector);
  * loss_exp = 1.0 in every vitg16 config, so the distance is plain L1.

    CUDA_VISIBLE_DEVICES=$G .../envs/cosmos/bin/python scripts/v21_score.py <input.json>
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

SIZE, PATCH, TUBELET, FRAMES, LEVELS = 384, 16, 2, 64, 4
GRID = SIZE // PATCH
TPS = GRID * GRID
DEPTH = FRAMES // TUBELET
# Fewer context lengths than before, because the context now needs its OWN encoder forward
# per masking rather than a gather from one full-clip encode.
CTXT = (4, 8, 12, 16, 20, 24, 28)


def read_clip(path):
    """mp4 -> (FRAMES, SIZE, SIZE, 3) uint8, letterboxed. Same convention as the rest of
    the harness: never centre-crop, a crop can carry the object out of frame."""
    import imageio.v2 as imageio
    from PIL import Image
    rd = imageio.get_reader(str(path))
    fr = [f[..., :3] for f in rd]
    rd.close()
    fr = np.stack(fr)
    fr = fr[np.linspace(0, len(fr) - 1, FRAMES).round().astype(int)]
    t, h, w, _ = fr.shape
    s = min(SIZE / w, SIZE / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    ox, oy = (SIZE - nw) // 2, (SIZE - nh) // 2
    out = np.zeros((t, SIZE, SIZE, 3), np.uint8)
    for i in range(t):
        out[i, oy:oy + nh, ox:ox + nw] = np.asarray(
            Image.fromarray(fr[i]).resize((nw, nh), Image.BILINEAR))
    return out


def strip(sd, *pref):
    out = {}
    for k, v in sd.items():
        for p in pref:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def main():
    import torch
    import torch.nn.functional as F
    from src.hub.backbones import _make_vjepa2_1_model

    torch.backends.cuda.matmul.allow_tf32 = False
    src = Path(sys.argv[1])
    blob = json.loads(src.read_text())
    clips = blob["clips"]

    # TWO encoders, as evals/intuitive_physics/eval.py does:
    #     h       = target_encoder(pieces, full_mask)     <- targets, full clip
    #     context = encoder(pieces, masks_enc)            <- context, MASKED input
    #     preds   = predictor(context, targets, masks_enc, masks_pred)
    # The first version of this script gathered the context out of one full-clip encode,
    # which is what HF's VJEPA2Model does. That leaks: a context token computed with full
    # attention has already seen the region it is being asked to predict. Measured on this
    # checkpoint, max |masked - gathered| = 0.043 on features whose signal is ~0.4%.
    enc_c, pred = _make_vjepa2_1_model(model_name=ARCH, img_size=SIZE,
                                       predictor_num_mask_tokens=8,
                                       n_output_distillation=LEVELS,
                                       return_all_tokens=True, pretrained=False)
    enc_t, _ = _make_vjepa2_1_model(model_name=ARCH, img_size=SIZE,
                                    predictor_num_mask_tokens=8,
                                    n_output_distillation=LEVELS,
                                    return_all_tokens=True, pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    for nm, mod, key in (("context encoder", enc_c, "encoder"),
                         ("target encoder", enc_t, "target_encoder"),
                         ("predictor", pred, "predictor")):
        r = mod.load_state_dict(strip(ck[key], "module.", "backbone."), strict=False)
        if r.missing_keys:
            raise RuntimeError(f"{nm}: {len(r.missing_keys)} missing, e.g. {r.missing_keys[:3]}")
        print(f"  {nm}: loaded ({len(r.unexpected_keys)} unexpected ignored)")
    del ck
    enc_c = enc_c.to("cuda:0").eval(); enc_t = enc_t.to("cuda:0").eval()
    pred = pred.to("cuda:0").eval()
    enc = enc_t
    E = enc.embed_dim
    print(f"encoder {E}  predictor out {LEVELS * E}  grid {GRID}x{GRID}  {DEPTH} slots")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

    def slot_tokens(a, b, sp=None):
        s = np.arange(TPS) if sp is None else np.asarray(sp, np.int64)
        return (np.arange(a, b)[:, None] * TPS + s[None, :]).reshape(-1)

    def pixels(frames):
        x = torch.as_tensor(frames, dtype=torch.float32).div_(255.0)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        return ((x - mean) / std).to("cuda:0")

    def encode(x, model, ids=None):
        with torch.no_grad():
            h = model(x, masks=None if ids is None else [ids], training=True)
        return h[0] if isinstance(h, (list, tuple)) else h

    def targets(h):
        return torch.cat([F.layer_norm(h[:, :, i * E:(i + 1) * E], (E,))
                          for i in range(LEVELS)], dim=2)

    def losses(x, h, per_slot=None):
        """h is the TARGET encode (full clip, target_encoder). The context is re-encoded
        per masking with the context encoder, which is the whole point of the fix."""
        hn, out = targets(h), []
        for n in CTXT:
            if not 0 < n < DEPTH:
                continue
            tgt = (slot_tokens(n, DEPTH) if per_slot is None else
                   np.concatenate([n_ * TPS + np.asarray(per_slot[n_], np.int64)
                                   for n_ in range(n, DEPTH)]))
            c = torch.as_tensor(slot_tokens(0, n), device=h.device).unsqueeze(0)
            t = torch.as_tensor(tgt, device=h.device).unsqueeze(0)
            ct = encode(x, enc_c, ids=c)                    # <- masked context encode
            with torch.no_grad():
                z = pred(ct, [c], [t])
            z = z[0] if isinstance(z, (list, tuple)) else z
            g = torch.gather(hn, 1, t.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
            out.append(float((z.float() - g.float()).abs().mean()))
        return np.asarray(out)

    rows = {}
    for i, (key, r) in enumerate(sorted(clips.items())):
        x = pixels(read_clip(r["clip"]))
        h = encode(x, enc_t)
        if i == 0:
            print(f"  token check: {h.shape[1]} tokens (expect {DEPTH * TPS}), "
                  f"width {h.shape[2]} (expect {LEVELS * E})", flush=True)
            assert h.shape[1] == DEPTH * TPS and h.shape[2] == LEVELS * E
        L = losses(x, h)
        O = losses(x, h, per_slot=r["object_patches"])
        rows[key] = {k: r[k] for k in ("probe", "param", "value", "ref") if k in r}
        rows[key].update(pair_with=r.get("pair_with"), pair_kind=r.get("pair_kind"),
                         max=float(L.max()), mean=float(L.mean()),
                         obj_max=float(O.max()), obj_mean=float(O.mean()),
                         obj_patches=r["median_patches"])
        print(f"  {key:<34} max {L.max():.5f}  obj_max {O.max():.5f}", flush=True)
    dst = Path("docs/results") / f"v21fix_{src.stem.replace('v21_input_', '')}.json"
    dst.write_text(json.dumps({"model": ARCH, "rows": rows}, indent=1))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
