"""L1 -- fetch Google Scanned Objects (Downs et al. 2022; CC BY 4.0) from Gazebo Fuel.

~1,030 scanned household objects at real scale, each a zip with meshes/model.obj, a texture
and an SDF. Into data/assets/gso/ (git-ignored): index.json (name, size, license) and one
directory per model. Resumable: finished models are skipped.

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    $L scripts/l1_fetch_gso.py [--workers 4]
"""
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DST = REPO / "data/assets/gso"
API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"


def get(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as exc:                            # network hiccup: back off and retry
            if k == tries - 1 or (isinstance(exc, urllib.error.HTTPError) and exc.code == 404):
                raise
            time.sleep(2 ** k)


def listing():
    out, page = [], 1
    while True:
        try:
            d = json.loads(get(f"{API}?per_page=100&page={page}", tries=2))
        except urllib.error.HTTPError as exc:               # Fuel answers 404 past the last page
            if exc.code == 404:
                return out
            raise
        if not d:
            return out
        out += [{"name": m["name"], "filesize": m["filesize"], "license": m.get("license_name")} for m in d]
        page += 1


def fetch(m):
    d = DST / m["name"]
    if (d / "meshes" / "model.obj").exists():
        return m["name"], "cached"
    raw = get(f"{API}/{urllib.parse.quote(m['name'])}.zip")
    d.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(raw)).extractall(d)
    return m["name"], "ok" if (d / "meshes" / "model.obj").exists() else "no-obj"


def main():
    DST.mkdir(parents=True, exist_ok=True)
    idx = DST / "index.json"
    models = json.loads(idx.read_text()) if idx.exists() else listing()
    idx.write_text(json.dumps(models, indent=0))
    print(f"{len(models)} models, {sum(m['filesize'] for m in models) / 1e9:.1f} GB", flush=True)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    status = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (name, st) in enumerate(ex.map(fetch, models)):
            status[name] = st
            if i % 50 == 0:
                print(f"  {i + 1}/{len(models)} {name}: {st}", flush=True)
    bad = {k: v for k, v in status.items() if v not in ("ok", "cached")}
    (DST / "fetch_status.json").write_text(json.dumps(status, indent=0))
    print(f"done: {sum(v in ('ok', 'cached') for v in status.values())} ok, {len(bad)} problems", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
