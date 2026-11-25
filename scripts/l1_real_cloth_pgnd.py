"""L1 -- PGND's robot cloth windows -> cloth episodes scored on observations (lsim.real.pgnd).

    L=/home/jooyeolyun/anaconda3/envs/lsim/bin/python
    OMP_NUM_THREADS=2 $L scripts/l1_real_cloth_pgnd.py [--procs 8] [--redo]

1. Every window's first cloud is unrolled (lsim.real.pgnd.layout; cached in outputs/lsim/pgnd_layouts.json).
2. Per recording (one cloth, one grasp: the grippers never open): the rest rectangle -- the median of its flattest
   windows' layouts -- and the table height (the median height of the cloth lying flat, less the teacher's resting
   height; recordings with no flat window take their day's).
3. Every window is converted (lsim.real.pgnd.convert_window: the grid fitted to its first cloud, tracked 10 frames,
   the held patches moved by the robot to frame 120, the observed clouds at frames 0 / 60 / 120) into
   data/real/cloth/pgnd/<cloth>_e<recording>_p<first frame>.{npz,json} (git-ignored: ~2 GB).
Writes docs/results/l1_real_cloth_pgnd.json (per recording: rest, table, windows; per window: the checks).
"""
import os

# CPU only, and no GPU at all: lsim.real.pgnd imports the teacher (warp), which initialises CUDA on every visible
# GPU. Initialised in the parent and forked into 32 workers, that wedged the NVIDIA driver's lock for the whole
# shared box (2026-10-22: every nvidia-smi on it hung). Hide the GPUs before any import; workers are spawned, not forked.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def _layout(path):
    import torch
    torch.set_num_threads(1)
    from lsim.real import pgnd
    w = pgnd.load_window(path)
    _, info = pgnd.layout(w["x"][0], w["tip"][0], calibrate=w["source"] not in pgnd.BOX)
    info.update({"path": str(path), "source": w["source"], "episode": w["episode"], "pivot": w["pivot"]})
    return info


def _convert(job):
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    from lsim.real import pgnd
    path, rest, table_z, nxt, nxt2 = job
    t0 = time.time()
    try:
        out, meta = pgnd.convert_window(Path(path), rest, table_z, None if nxt is None else Path(nxt),
                                        None if nxt2 is None else Path(nxt2), log=lambda *a: None)
    except Exception as e:                                              # noqa: BLE001 -- report, keep going
        return {"path": path, "error": repr(e)[:300]}
    meta["seconds"] = time.time() - t0
    pgnd.save_episode(out, meta)
    return {"path": path, "case": meta["case"], "split": meta["split"], "box": meta["box"], "layout": meta["layout"],
            "quality": meta["quality"], "held": meta["held"], "clouds": meta["clouds"], "seconds": meta["seconds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--redo", action="store_true", help="convert windows already converted again")
    ap.add_argument("--limit", type=int, default=0, help="convert only this many windows (a smoke test)")
    a = ap.parse_args()
    from lsim.real import pgnd
    windows = pgnd.pgnd_windows()
    chains = pgnd.recordings(windows)
    cache = REPO / "outputs/lsim/pgnd_layouts.json"
    if cache.exists():
        infos = json.loads(cache.read_text())
    else:
        t0 = time.time()
        with mp.get_context("spawn").Pool(a.procs) as p:
            infos = p.map(_layout, [str(w) for w in windows], chunksize=2)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(infos))
        print(f"[layouts] {len(infos)} windows in {time.time() - t0:.0f} s", flush=True)
    by_rec = {}
    for i in infos:
        by_rec.setdefault((i["source"], i["episode"]), []).append(i)
    rests = {k: pgnd.recording_rest(v) for k, v in by_rec.items()}
    day = {}
    for (src, ep), r in rests.items():
        if r["table_z"] is not None:
            day.setdefault(src[:4], []).append(r["table_z"])
    for (src, ep), r in rests.items():
        if r["table_z"] is None:
            r["table_z"] = float(np.median(day[src[:4]]))
            r["table_from"] = "day"
        print(f"[rest] {pgnd.SHORT[src]} e{ep}: {len(by_rec[(src, ep)])} windows, rest {100 * r['W']:.1f} x {100 * r['H']:.1f} cm "
              f"(median of {r['n_windows']}), table at raw z {1e3 * r['table_z']:.1f} mm{' (the day' + chr(39) + 's)' if r.get('table_from') else ''}",
              flush=True)
    jobs = []
    for w in windows:
        m = np.loadtxt(w / "meta.txt").astype(int)
        r = rests[(w.parts[-3], int(m[0]))]
        n1, n2 = pgnd.next_windows(w, chains)
        jobs.append((str(w), {"W": r["W"], "H": r["H"]}, r["table_z"], None if n1 is None else str(n1),
                     None if n2 is None else str(n2)))
    if a.limit:
        jobs = jobs[:: max(1, len(jobs) // a.limit)][: a.limit]

    def case_of(path):
        m = np.loadtxt(Path(path) / "meta.txt").astype(int)
        return f"{pgnd.SHORT[Path(path).parts[-3]]}_e{m[0]}_p{m[1]:05d}"

    done = {p.stem for p in pgnd.OUT.glob("*.json")} if not a.redo else set()
    todo = [j for j in jobs if case_of(j[0]) not in done]
    print(f"[convert] {len(todo)} windows to convert ({len(jobs) - len(todo)} already done)", flush=True)
    t0 = time.time()
    bad = []
    with mp.get_context("spawn").Pool(a.procs) as p:  # few processes: the raw windows and the episodes are on the NAS, and
        for i, r in enumerate(p.imap_unordered(_convert, todo)):        # 32 readers stalled everything on it
            if "error" in r:
                bad.append(r)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"[convert] {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    good = []
    for j in sorted(pgnd.OUT.glob("*.json")):
        m = json.loads(j.read_text())
        good.append({"path": m["pgnd_source"] + "/" + m["pgnd_window"], "case": m["case"], "split": m["split"], "box": m["box"],
                     "layout": m["layout"], "quality": m["quality"], "held": m["held"], "clouds": m["clouds"],
                     "seconds": m.get("seconds")})

    def stat(key, sub=None):
        v = np.array([r["quality"][key] for r in good if key in r["quality"] and (sub is None or sub(r))], float)
        return {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": int(len(v))} if len(v) else None

    summary = {"recordings": {f"{pgnd.SHORT[s]}_e{e}": dict(rests[(s, e)], windows=len(by_rec[(s, e)]),
                                                            split="test" if (s, e) in pgnd.TEST else "train", box=s in pgnd.BOX)
                              for (s, e) in sorted(rests)},
               "converted": len(good), "errors": bad,
               "cloud0_to_mesh_mm": stat("cloud0_to_mesh_mm"), "cloud0_to_mesh_p90_mm": stat("cloud0_to_mesh_p90_mm"),
               "carried_to_cloud60_mm": stat("carried_to_cloud60_mm"),
               "strain_p99": stat("strain_p99"), "known0": stat("known0"),
               "res_last_tracked_mm": {"median": float(np.median([r["quality"]["res_mm"][-1] for r in good])) if good else None},
               "windows": sorted(good, key=lambda r: r["case"])}
    out = REPO / "docs/results/l1_real_cloth_pgnd.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"[done] {len(good)} converted, {len(bad)} failed; first cloud to mesh {summary['cloud0_to_mesh_mm']}; "
          f"carried points to the observed cloud at frame 60 {summary['carried_to_cloud60_mm']} -> {out}", flush=True)


if __name__ == "__main__":
    main()
