"""Pick a GPU that is actually free, and never sit on one.

The box is shared and credit-metered, so a job may occupy a card only while it is
computing. This polls `nvidia-smi` rather than torch, so the sim env (no torch) and
the Blender render stage can use it too.

`wait_for_gpu` returns a PHYSICAL index. Callers set CUDA_VISIBLE_DEVICES to it before CUDA
initialises (see scripts/l0_teacher_gate.py) and let the card go when the stage ends.
"""
import subprocess
import time

QUERY = "index,memory.total,memory.used,utilization.gpu"


def gpu_state():
    """[(index, free_gib, util_pct)] for every visible GPU."""
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        idx, total, used, util = (x.strip() for x in line.split(","))
        rows.append((int(idx), (int(total) - int(used)) / 1024.0, int(util)))
    return rows


def free_gpu(min_free_gib=8.0, max_util=40):
    """Least-utilised GPU meeting both bars, or None."""
    ok = [r for r in gpu_state() if r[1] >= min_free_gib and r[2] <= max_util]
    if not ok:
        return None
    ok.sort(key=lambda r: (r[2], -r[1]))
    return ok[0][0]


def wait_for_gpu(min_free_gib=8.0, max_util=40, poll_s=60, timeout_s=None, log=print):
    """Block until a GPU is free. Returns its physical index.

    Deliberately has no fallback to "least bad" -- running Cycles or a 8192-token
    ViT-L against a card at 100% both distorts our own timings and steals from the
    job already there.
    """
    t0 = time.time()
    said = False
    while True:
        g = free_gpu(min_free_gib, max_util)
        if g is not None:
            log(f"[gpu] using cuda:{g}")
            return g
        if not said:
            log(f"[gpu] waiting for a card with >={min_free_gib:.0f} GiB free and <={max_util}% util")
            log("[gpu] " + " | ".join(f"{i}: {f:.1f}G {u}%" for i, f, u in gpu_state()))
            said = True
        if timeout_s is not None and time.time() - t0 > timeout_s:
            raise TimeoutError(f"no free GPU after {timeout_s:.0f}s")
        time.sleep(poll_s)


if __name__ == "__main__":
    for i, f, u in gpu_state():
        print(f"gpu {i}: {f:6.1f} GiB free, {u:3d}% util")
    print("free ->", free_gpu())
