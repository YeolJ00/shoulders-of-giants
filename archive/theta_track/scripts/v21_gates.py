"""V0 gates for the 2.1 ViT-g adapter. Standalone (Meta's `src` shadows ours).

An unvalidated scoring model cannot adjudicate anything -- rule 5. These are the same
gates V-JEPA 2 had to pass:
  1 DETERMINISM  same clip twice, identical float
  2 PREDICTS     beats predicting zero against the layer-normed target
  3 ORDERING     a frame-shuffled natural clip must read as MORE surprising than the real
                 one. This is the decisive one: if it fails, the integration is wrong, not
                 the model.
"""
import sys
from pathlib import Path
import numpy as np

VJ = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_repo"
CKPT = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_1_vitg_384.pt"
sys.path.insert(0, VJ)
SIZE, PATCH, TUBELET, FRAMES, LEVELS = 384, 16, 2, 64, 4
GRID = SIZE // PATCH; TPS = GRID * GRID; DEPTH = FRAMES // TUBELET
REFS = ["outputs/surprise/ref_kinetics_archery.mp4", "outputs/surprise/ref_kinetics_bowling.mp4"]


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


def strip(sd, *p):
    o = {}
    for k, v in sd.items():
        for q in p:
            if k.startswith(q):
                k = k[len(q):]
        o[k] = v
    return o


def main():
    import torch, torch.nn.functional as F
    from src.hub.backbones import _make_vjepa2_1_model
    enc, pred = _make_vjepa2_1_model(model_name="vjepa2_1_vit_giant_384", img_size=SIZE,
                                     predictor_num_mask_tokens=8, n_output_distillation=LEVELS,
                                     return_all_tokens=True, pretrained=False)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict(strip(ck["target_encoder"], "module.", "backbone."), strict=False)
    pred.load_state_dict(strip(ck["predictor"], "module.", "backbone."), strict=False)
    del ck
    enc, pred = enc.to("cuda:0").eval(), pred.to("cuda:0").eval()
    E = enc.embed_dim
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)

    def encode(f):
        x = torch.as_tensor(f, dtype=torch.float32).div_(255.).permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            h = enc(((x - mean) / std).to("cuda:0"), masks=None, training=True)
        return h[0] if isinstance(h, (list, tuple)) else h

    def tgt_norm(h):
        return torch.cat([F.layer_norm(h[:, :, i*E:(i+1)*E], (E,)) for i in range(LEVELS)], 2)

    def score(f, ctxt=range(2, 31, 2)):
        h = encode(f); hn = tgt_norm(h); vals, zeros = [], []
        for n in ctxt:
            c = torch.arange(0, n*TPS, device=h.device).unsqueeze(0)
            t = torch.arange(n*TPS, DEPTH*TPS, device=h.device).unsqueeze(0)
            ct = torch.gather(h, 1, c.unsqueeze(-1).expand(-1, -1, h.size(-1)))
            with torch.no_grad():
                z = pred(ct, [c], [t])
            z = z[0] if isinstance(z, (list, tuple)) else z
            g = torch.gather(hn, 1, t.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
            vals.append(float((z.float()-g.float()).abs().mean()))
            zeros.append(float(g.float().abs().mean()))
        return np.asarray(vals), np.asarray(zeros)

    ok = []
    f = read_clip(REFS[0])
    a, _ = score(f); b, _ = score(f)
    p1 = np.array_equal(a, b)
    print(f"1 DETERMINISM   {'PASS' if p1 else 'FAIL'}  {a.max():.9f} vs {b.max():.9f}"); ok.append(p1)
    v, z = score(f)
    p2 = v.mean() < z.mean()
    print(f"2 PREDICTS      {'PASS' if p2 else 'FAIL'}  model {v.mean():.5f} vs zero "
          f"{z.mean():.5f}  ({100*(1-v.mean()/z.mean()):+.1f}% better)"); ok.append(p2)
    print("3 ORDERING")
    res = []
    for r in REFS:
        if not Path(r).exists():
            continue
        f = read_clip(r); rng = np.random.default_rng(0)
        real, _ = score(f); shuf, _ = score(f[rng.permutation(len(f))])
        stat, ss = real.max(), shuf.max()
        res.append(ss > stat)
        print(f"    {Path(r).stem:<26} real {stat:.5f}  shuffled {ss:.5f}  "
              f"{100*(ss/stat-1):+.2f}%   {'ok' if ss>stat else 'WRONG SIGN'}")
    p3 = all(res)
    print(f"   {'PASS' if p3 else 'FAIL'}"); ok.append(p3)
    print(f"\nV-JEPA 2.1 adapter: {'PASS' if all(ok) else 'FAIL'} ({sum(ok)}/{len(ok)})")


if __name__ == "__main__":
    main()
