"""L1 -- process fetched scans into learner / collision meshes and geometry labels.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_process_assets.py [--workers 16] [--limit N]

Reads data/assets/gso/<name>/meshes/model.obj, writes data/assets/proc/gso/<name>.npz and
data/assets/proc/gso/index.json. Resumable. CPU only (the box is shared: 16 workers of 64).
"""
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data/assets/gso"
DST = REPO / "data/assets/proc/gso"


def work(name):
    out = DST / f"{name}.npz"
    try:
        if not out.exists():
            from lsim.data.assets import process
            t0 = time.time()
            d = process(SRC / name / "meshes" / "model.obj", name)
            np.savez_compressed(out, **d)
            dt = time.time() - t0
        else:
            dt = 0.0
        d = np.load(out)
        return {"name": name, "ok": True, "seconds": round(dt, 1), "watertight": bool(d["watertight"]),
                "extents": d["extents"].tolist(), "volume": float(d["volume"]), "n_learner": int(len(d["learner_v"])),
                "edge": float(d["learner_edge"]), "n_collide_faces": int(len(d["collide_f"])),
                "convex_n": int(d["convex_n"])}
    except Exception as exc:
        return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "trace": traceback.format_exc(limit=2)[-400:]}


def main():
    DST.mkdir(parents=True, exist_ok=True)
    names = sorted(p.parent.parent.name for p in SRC.glob("*/meshes/model.obj"))
    if "--limit" in sys.argv:
        names = names[:int(sys.argv[sys.argv.index("--limit") + 1])]
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 16
    rows = []
    with Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, names)):
            rows.append(r)
            if i % 50 == 0 or not r["ok"]:
                print(f"  {i + 1}/{len(names)} {r['name']}: {'ok' if r['ok'] else r['error']}", flush=True)
    rows.sort(key=lambda r: r["name"])
    (DST / "index.json").write_text(json.dumps(rows, indent=0))
    ok = [r for r in rows if r["ok"]]
    print(f"done: {len(ok)}/{len(rows)} ok, watertight {sum(r['watertight'] for r in ok)}, "
          f"learner verts median {np.median([r['n_learner'] for r in ok]):.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
