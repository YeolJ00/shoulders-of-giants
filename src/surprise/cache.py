"""Score cache, keyed by the clip's bytes and the scoring configuration.

Rule 7: renders and scoring calls are the cost centres. Rule 5: any change to the
scoring model invalidates calibration -- so the model id, layer, dtype and context
length are all part of the key. Change one and you get a miss, not a stale number
silently reused across a recalibration.

Keyed on the CLIP BYTES rather than on the theta that produced it, deliberately: a
re-render of identical physics with a different Cycles seed is a different clip, and
the noise-floor measurement is exactly the spread over those.

Follows src/render/clip.py::theta_key -- sha1 over sorted-key JSON, truncated.
"""
import hashlib
import json
import os
from pathlib import Path

from .clips import clip_hash

ROOT = Path(os.environ.get("SURPRISE_CACHE", "outputs/surprise/cache"))


def config_key(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


class ScoreCache:
    """One JSON file per configuration; entries keyed by clip hash."""

    def __init__(self, cfg, root=ROOT):
        self.cfg = dict(cfg)
        self.key = config_key(self.cfg)
        self.path = Path(root) / f"{self.key}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            blob = json.loads(self.path.read_text())
            self.entries = blob.get("entries", {})
        else:
            self.entries = {}
            self.flush()

    def get(self, clip):
        return self.entries.get(clip_hash(clip))

    def put(self, clip, value):
        self.entries[clip_hash(clip)] = value
        self.flush()
        return value

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"config": self.cfg, "entries": self.entries}, indent=1))
        tmp.replace(self.path)
