"""V-JEPA 2.1 (non-distilled ViT-g/ViT-G) as a surprise model.

Why this checkpoint and not the HuggingFace one: the ported `vjepa2.1-vit-base-384` is
`..._dist_vitG_...`, a ViT-B DISTILLED from ViT-G, whose predictor projects into the
teacher's 1664-d space and so cannot be compared against its own 768-d encoder. The ViT-g
and ViT-G checkpoints carry no `dist` tag because they ARE the teachers: verified by
loading, `encoder.embed_dim = 1408` and `predictor.proj.out_features = 5632 = 4 x 1408`,
i.e. four Deep-Self-Supervision levels at the encoder's own width. Self-consistent, which
is what surprise requires.

The target recipe is copied from the training code (app/vjepa_2_1/train.py::forward_target),
not inferred. Two details would have been wrong if guessed:

  * the 5632-wide hierarchical feature is split into four 1408 chunks and EACH is
    layer-normed separately -- not the whole vector;
  * the loss is mean(|z - h| ** loss_exp) / loss_exp with loss_exp = 1.0 in every
    vitg16 config, i.e. plain L1.

Geometry: 384 / patch 16 -> 24 x 24 = 576 spatial patches, 2.25x the density of the
256-crop V-JEPA 2 models, which is the reason to try it on scenes where the object is
small in frame.
"""
import numpy as np

REPO = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_repo"
CKPT = "/home/nas5/jooyeolyun/repos/simulation-assestization/models/vjepa2_1_vitg_384.pt"
ARCH = "vjepa2_1_vit_giant_384"
LEVELS = 4


def _strip(sd, *prefixes):
    out = {}
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


class Surprise21:
    def __init__(self, ckpt=CKPT, arch=ARCH, device="cuda:0", dtype="float32",
                 deterministic=True):
        import sys
        import torch
        sys.path.insert(0, REPO)
        from src.hub.backbones import _make_vjepa2_1_model

        self.torch = torch
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device, self.dtype = device, getattr(torch, dtype)
        self.model_id = arch
        enc, pred = _make_vjepa2_1_model(model_name=arch, img_size=384,
                                         predictor_num_mask_tokens=8,
                                         n_output_distillation=LEVELS,
                                         return_all_tokens=True, pretrained=False)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        e = enc.load_state_dict(_strip(ck["target_encoder"], "module.", "backbone."),
                                strict=False)
        p = pred.load_state_dict(_strip(ck["predictor"], "module.", "backbone."),
                                 strict=False)
        # loud, because a silently half-loaded model would still produce plausible numbers
        for nm, r in (("encoder", e), ("predictor", p)):
            if r.missing_keys:
                raise RuntimeError(f"{nm}: {len(r.missing_keys)} missing keys, "
                                   f"e.g. {r.missing_keys[:3]}")
        self.enc = enc.to(device, self.dtype).eval()
        self.pred = pred.to(device, self.dtype).eval()
        self.embed = enc.embed_dim
        self.size, self.patch, self.tubelet, self.frames = 384, 16, 2, 64
        self.grid = self.size // self.patch
        self.tps = self.grid * self.grid
        self.clip_frames = self.frames
        self.depth = self.frames // self.tubelet
        self.n_tokens = self.depth * self.tps
        self._checked = False

    def config(self):
        return {"model": self.model_id, "ckpt": "vjepa2_1_vitg_384", "dtype": str(self.dtype),
                "size": self.size, "clip_frames": self.clip_frames, "levels": LEVELS,
                "target_norm": "per-chunk layer_norm", "metric": "l1"}

    def slot_tokens(self, s0, s1, spatial=None):
        sp = np.arange(self.tps) if spatial is None else np.asarray(spatial, np.int64)
        return (np.arange(s0, s1)[:, None] * self.tps + sp[None, :]).reshape(-1)

    # ------------------------------------------------------------------ pixels

    def _pixels(self, frames):
        torch = self.torch
        x = torch.as_tensor(np.asarray(frames), dtype=torch.float32).div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        x = x.permute(3, 0, 1, 2).unsqueeze(0)            # (1, C, T, H, W)
        x = (x - mean) / std
        return x.to(self.device, self.dtype)

    # ------------------------------------------------------------- enc / pred

    def encode(self, frames):
        """Hierarchical context features, (1, N, 4*embed)."""
        torch = self.torch
        x = self._pixels(frames)
        with torch.no_grad():
            h = self.enc(x, masks=None, training=True)
        h = h[0] if isinstance(h, (list, tuple)) else h
        if not self._checked:
            if h.shape[1] != self.n_tokens:
                raise RuntimeError(f"token count {h.shape[1]} != {self.n_tokens}")
            if h.shape[2] != LEVELS * self.embed:
                raise RuntimeError(f"width {h.shape[2]} != {LEVELS * self.embed}")
            self._checked = True
        return h

    def targets(self, h):
        """train.py::forward_target -- each level layer-normed SEPARATELY, then concat."""
        F = self.torch.nn.functional
        d = self.embed
        return self.torch.cat(
            [F.layer_norm(h[:, :, i * d:(i + 1) * d], (d,)) for i in range(LEVELS)], dim=2)

    def predict(self, h, ctx_ids, tgt_ids):
        torch = self.torch
        ctx = torch.as_tensor(np.asarray(ctx_ids, np.int64), device=h.device).unsqueeze(0)
        tgt = torch.as_tensor(np.asarray(tgt_ids, np.int64), device=h.device).unsqueeze(0)
        ctx_tok = torch.gather(h, 1, ctx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        with torch.no_grad():
            z = self.pred(ctx_tok, [ctx], [tgt])
        z = z[0] if isinstance(z, (list, tuple)) else z
        hn = self.targets(h)
        g = torch.gather(hn, 1, tgt.unsqueeze(-1).expand(-1, -1, hn.size(-1)))
        return z.float(), g.float()

    def prefix_losses(self, h, ctxt, per_slot=None):
        out = []
        for n in ctxt:
            if not 0 < n < self.depth:
                continue
            tgt = (self.slot_tokens(n, self.depth) if per_slot is None else
                   np.concatenate([s * self.tps + per_slot[s]
                                   for s in range(n, self.depth)]))
            z, g = self.predict(h, self.slot_tokens(0, n), tgt)
            out.append(float((z - g).abs().mean()))
        return np.asarray(out)
