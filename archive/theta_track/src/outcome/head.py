"""The SSv2 outcome head: clip -> distribution over 174 physical outcomes.

Model choice, verified against the artifacts rather than recalled:
  * There is NO V-JEPA 2.1 SSv2 head. Meta ships `configs/eval_2_1/vitG-384/ssv2.yaml`
    but no trained probe; training one needs 16 nodes x 8 GPUs.
  * The V-JEPA 2 heads ARE released: ViT-L/16 -> 73.7%, ViT-g/16_384 -> 77.3%. The widely
    quoted "77.7 on 2.1" is neither the right number nor the right model.
  * `frames_per_clip = 64` on the ViT-g head matches our existing clip length exactly.

Unlike the surprise work there is NO MASKING in a classification forward, so the
HuggingFace path is safe here -- the context leak that made HF unusable for surprise
(context tokens gathered from a full-clip encode had already attended to the region being
predicted) simply cannot arise. One call gives encoder + attentive probe + 174 logits.

Conventions follow src/surprise/: module constants, a `device=` argument, fp32, a
guard-once check that the video actually reached the model, and hand-rolled imageio
decoding -- the framework's own video loader is broken in this environment and fails by
returning nothing, which is how the judge once scored text alone.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HEAD = "facebook/vjepa2-vitg-fpc64-384-ssv2"
FRAMES, SIZE = 64, 384
CACHE = Path(os.environ.get("OUTCOME_CACHE", "outputs/outcome/cache"))


def read_frames(path):
    import imageio.v2 as imageio
    rd = imageio.get_reader(str(path))
    fr = [f[..., :3] for f in rd]
    rd.close()
    if not fr:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(fr)


def prepare(path, n=FRAMES, size=SIZE):
    """mp4 -> (n, size, size, 3) uint8, letterboxed.

    Letterbox rather than centre-crop, for the same reason as everywhere else here: a crop
    can carry the object out of frame, and where the object went is the entire observable.
    """
    from PIL import Image
    fr = read_frames(path)
    fr = fr[np.linspace(0, len(fr) - 1, n).round().astype(int)]
    t, h, w, _ = fr.shape
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out = np.zeros((t, size, size, 3), np.uint8)
    for i in range(t):
        out[i, oy:oy + nh, ox:ox + nw] = np.asarray(
            Image.fromarray(fr[i]).resize((nw, nh), Image.BILINEAR))
    return out


def clip_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class OutcomeHead:
    """One clip at a time, no batching -- the shape src/judge and src/surprise both use."""

    def __init__(self, model_id=HEAD, device="cuda:0", dtype="float32", deterministic=True):
        import torch
        from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

        self.torch = torch
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model_id, self.device = model_id, device
        self.dtype = getattr(torch, dtype)
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = VJEPA2ForVideoClassification.from_pretrained(
            model_id, dtype=self.dtype).to(device).eval()
        c = self.model.config
        self.labels = [c.id2label[str(i)] if str(i) in c.id2label else c.id2label[i]
                       for i in range(c.num_labels)]
        self.frames, self.size = int(c.frames_per_clip), int(c.crop_size)
        self._checked = False

    def config(self):
        return {"model": self.model_id, "dtype": str(self.dtype),
                "frames": self.frames, "size": self.size, "labels": len(self.labels)}

    def probs(self, frames):
        """(FRAMES, SIZE, SIZE, 3) uint8 -> (174,) float64 probability vector."""
        torch = self.torch
        px = self.processor(list(frames), return_tensors="pt")["pixel_values_videos"]
        px = px.to(self.device, self.dtype)
        with torch.no_grad():
            logits = self.model(pixel_values_videos=px).logits
        if not self._checked:
            # the judge silently scored TEXT ALONE for a whole calibration run because a
            # broken decoder handed it no video; this raises instead of warning
            if tuple(px.shape) != (1, self.frames, 3, self.size, self.size):
                raise RuntimeError(f"video did not reach the model: {tuple(px.shape)}")
            if logits.shape[-1] != len(self.labels):
                raise RuntimeError(f"{logits.shape[-1]} logits vs {len(self.labels)} labels")
            self._checked = True
        return torch.softmax(logits.float(), -1)[0].cpu().numpy().astype(np.float64)

    def score_clip(self, path, use_cache=True):
        key = clip_hash(path)
        f = CACHE / f"{key}.json"
        if use_cache and f.exists():
            return np.asarray(json.loads(f.read_text())["probs"], np.float64)
        p = self.probs(prepare(path, self.frames, self.size))
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"clip": str(path), "model": self.model_id,
                                 "probs": p.tolist()}))
        return p

    def top(self, p, k=5):
        idx = np.argsort(-p)[:k]
        return [(self.labels[i], float(p[i])) for i in idx]
