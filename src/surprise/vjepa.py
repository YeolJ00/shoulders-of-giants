"""V-JEPA 2 surprise: predict what happens next, in representation space.

    surprise = L1( predictor(context) , layer_norm(encoder(target)) )

which is V-JEPA's own pretraining objective. Nothing is generated and nothing is
prompted; the model only reacts to a motion the simulator made.

Three things here were settled by measurement, not by assumption. Each was a wrong
first guess (scripts/v0_surprise.py re-checks all of them at runtime):

**The target must be layer-normed.** V-JEPA's loss is `smooth_l1(z, F.layer_norm(h))`
-- normalised over the feature dim, no affine. `VJEPA2Model` returns the target
*raw*, and the raw encoder output has mean|.| = 2.15 while the predictor emits 0.32,
so an un-normalised L1 is a large constant plus a rounding error. Measured against
the raw target, a STATIC clip scored as more surprising than a real one and every
variant landed within 2.4% of 2.03. Normalised, the ordering is right.

**The context is a causal prefix that slides, not one long jump.** CLAUDE.md's V0
sketch was context = first K frames, target = all the rest. Predicting 48 frames
from 16 is far outside anything the predictor was trained on, and it shows: that
split separates a real clip from a frame-shuffled one by +0.12%. Sliding a short
horizon (predict the next 2 frames, from everything before them) gives +2.54% on
the same clips. Trained-regime spatial blocks give +1.49%.

**Token index -> (t, h, w).** Patch embedding is `Conv3d(...).flatten(2).transpose(1,2)`
and `VJEPA2RopeAttention._get_frame_pos` reads `t = n // (GRID*GRID)`. The masks are
INDEX tensors, not booleans -- `VJEPA2Predictor.forward` gathers the encoder states at
`context_mask`, appends learned mask tokens at `target_mask`, and derives its rope
positions from those same indices.

The predictor consumes the encoder's LAST hidden state, so CLAUDE.md's "middle encoder
layers" knob cannot apply to the predictor target -- feeding the predictor a layer it
never saw is off-distribution. `layer` therefore selects the target representation only
for a secondary, predictor-free variant.

fp32 throughout, deliberately. The judge taught this: a bf16 head at these magnitudes
quantises in steps far coarser than the differences being measured. Speed is not the
constraint -- a Cycles render costs ~35 s and a scored clip costs ~2 s.
"""
import numpy as np

MODEL = "facebook/vjepa2-vitl-fpc64-256"

FRAMES = 64          # fixed by the rope grid; see clips.py
SIZE = 256
PATCH = 16
TUBELET = 2
GRID = SIZE // PATCH                 # 16 patches per side
TOKENS_PER_SLOT = GRID * GRID        # 256 tokens per 2-frame slot
DEPTH = FRAMES // TUBELET            # 32 temporal slots
N_TOKENS = DEPTH * TOKENS_PER_SLOT   # 8192

# The causal curve starts once there is enough history to predict from, and each step
# predicts one slot (2 frames) ahead. Both are config, and both are part of the cache key.
CAUSAL_START = 4
HORIZON = 1

ALL_TOKENS = np.arange(N_TOKENS)


def slot_tokens(s0, s1, spatial=None):
    """Token ids for temporal slots [s0, s1), optionally only at the given patches."""
    sp = np.arange(TOKENS_PER_SLOT) if spatial is None else np.asarray(spatial, np.int64)
    return (np.arange(s0, s1)[:, None] * TOKENS_PER_SLOT + sp[None, :]).reshape(-1)


class SurpriseModel:
    """One clip at a time, no batching -- the same shape as src/judge."""

    def __init__(self, model_id=MODEL, device="cuda:0", dtype="float32", deterministic=True):
        import torch
        from transformers import AutoModel, AutoVideoProcessor

        self.torch = torch
        if deterministic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        self.model_id = model_id
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, dtype=self.dtype, attn_implementation="sdpa"
        ).to(device).eval()
        self._checked = False

    def config(self):
        """Everything that invalidates a cached score (rule 5)."""
        return {"model": self.model_id, "dtype": str(self.dtype), "frames": FRAMES,
                "size": SIZE, "causal_start": CAUSAL_START, "horizon": HORIZON,
                "target_norm": "layer_norm", "metric": "l1"}

    # ---------------------------------------------------------------- encoding

    def _pixels(self, frames):
        px = self.processor(list(frames), return_tensors="pt")["pixel_values_videos"]
        return px.to(self.device, self.dtype)

    def encode(self, frames, layer=None):
        """(FRAMES, SIZE, SIZE, 3) uint8 -> (1, N_TOKENS, 1024) encoder tokens.

        The guard-once check is the src/judge pattern: the judge silently scored text
        alone for a whole calibration run because a broken decoder handed it no video.
        A wrong token count is the same class of failure -- every rope position quietly
        off -- so it raises rather than warns.
        """
        torch = self.torch
        px = self._pixels(frames)
        with torch.no_grad():
            out = self.model(pixel_values_videos=px, skip_predictor=True,
                             output_hidden_states=layer is not None)
        toks = out.hidden_states[layer] if layer is not None else out.last_hidden_state
        if not self._checked:
            if tuple(px.shape) != (1, FRAMES, 3, SIZE, SIZE):
                raise RuntimeError(f"video did not reach the model as expected: {tuple(px.shape)}")
            if toks.shape[1] != N_TOKENS:
                raise RuntimeError(
                    f"token count {toks.shape[1]} != {N_TOKENS}; the rope position grid is "
                    f"built from frames_per_clip and would be silently wrong")
            self._checked = True
        return toks

    # --------------------------------------------------------------- predicting

    def predict(self, tokens, ctx_ids, tgt_ids):
        """Predicted vs actual (layer-normed) representations at tgt_ids."""
        torch = self.torch
        ctx = torch.as_tensor(np.asarray(ctx_ids, np.int64), device=tokens.device).unsqueeze(0)
        tgt = torch.as_tensor(np.asarray(tgt_ids, np.int64), device=tokens.device).unsqueeze(0)
        with torch.no_grad():
            pred = self.model.predictor(
                encoder_hidden_states=tokens, context_mask=[ctx], target_mask=[tgt]
            ).last_hidden_state.float()
        target = torch.gather(
            tokens, 1, tgt.unsqueeze(-1).expand(-1, -1, tokens.size(-1))).float()
        target = torch.nn.functional.layer_norm(target, (target.size(-1),))
        return pred, target

    def distance(self, pred, target):
        return {"l1": float((pred - target).abs().mean()),
                "smooth_l1": float(self.torch.nn.functional.smooth_l1_loss(pred, target)),
                "n_tokens": int(pred.shape[1])}

    # -------------------------------------------------------------- the readings

    def causal_curve(self, tokens, spatial=None, start=CAUSAL_START, horizon=HORIZON):
        """Surprise per temporal slot: predict slot s from everything before it.

        When `spatial` is given the target is only the object's patches at slot s, and
        the SAME-TIME background stays in the context -- so the model is told the camera
        did not move and is asked only where the object went. That is the observable we
        actually care about; a full-slot target averages it against a static table.
        """
        out = []
        for s in range(start, DEPTH - horizon + 1):
            tgt = slot_tokens(s, s + horizon, spatial)
            if spatial is None:
                ctx = slot_tokens(0, s)
            else:
                ctx = np.setdiff1d(slot_tokens(0, s + horizon), tgt)
            p, g = self.predict(tokens, ctx, tgt)
            out.append(self.distance(p, g)["l1"])
        return np.asarray(out)

    def block(self, tokens, spatial, s0=0, s1=DEPTH):
        """The regime V-JEPA was pretrained on: hold out a spatial block over time."""
        tgt = slot_tokens(s0, s1, spatial)
        p, g = self.predict(tokens, np.setdiff1d(ALL_TOKENS, tgt), tgt)
        return self.distance(p, g)

    def score(self, frames, object_spatial=None, layer=None, want_block=True):
        """Every reading this harness produces, from ONE encoder pass."""
        toks = self.encode(frames, layer=layer)
        c = self.causal_curve(toks)
        out = {"causal": {"mean": float(c.mean()), "max": float(c.max()),
                          "curve": [float(x) for x in c]}}
        if object_spatial is not None and len(object_spatial):
            o = self.causal_curve(toks, spatial=object_spatial)
            out["causal_object"] = {"mean": float(o.mean()), "max": float(o.max()),
                                    "curve": [float(x) for x in o]}
            if want_block:
                out["object_tube"] = self.block(toks, object_spatial, s0=DEPTH // 4)
        return out
