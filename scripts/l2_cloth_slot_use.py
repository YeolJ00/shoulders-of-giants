"""Does the cloth model's rollout respond to its material slots? c1_test, 32 episodes, 60 frames from frame 7:
error vs teacher with the true material given / hidden / another episode's material given, and how far the
given-true and given-swapped rollouts are apart."""
import sys, copy, glob, json
import numpy as np, torch
sys.path.insert(0, "/home/nas5/jooyeolyun/repos/simulation-assestization/scripts")
from l2_real_cloth_pgnd_eval import load_model
from lsim.train.cloth_train import _load, rollout
N = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
HID = ("density", "stretch", "bend", "mu")
R = "/home/nas5/jooyeolyun/repos/simulation-assestization/"
eps = _load(sorted(glob.glob(R + "data/datasets/c1_test/shard_*.npz")), set())
rng = np.random.default_rng(0)
eps = [eps[i] for i in rng.choice(len(eps), 200, replace=False) if eps[i].n_frames > 70][:32]
for ck in sys.argv[1:]:
    m = load_model(R + ck, "cuda")
    rows = []
    for i, e in enumerate(eps):
        o = eps[(i + 7) % len(eps)]
        sw = copy.copy(e); sw.c = dict(e.c); 
        for k in ("Y", "B", "w", "mu", "bending_length", "nu"):
            if k in o.c: sw.c[k] = o.c[k]
        with torch.no_grad():
            pg, tr = rollout(m, e, 7, 60, hidden=())[:2]
            ph = rollout(m, e, 7, 60, hidden=HID)[0]
            ps = rollout(m, sw, 7, 60, hidden=())[0]
        f = ~np.asarray(e.held[7 + 59]) if hasattr(e, "held") else slice(None)
        err = lambda p: float(1e3 * np.median(np.linalg.norm(N(p[-1]) - N(tr[-1]), axis=-1)))
        rows.append((err(pg), err(ph), err(ps), float(1e3 * np.median(np.linalg.norm(N(pg[-1]) - N(ps[-1]), axis=-1))),
                     float(1e3 * np.median(np.linalg.norm(N(pg[-1]) - N(ph[-1]), axis=-1)))))
    a = np.median(np.array(rows), 0)
    print(f"{ck}: 2 s error vs teacher  given {a[0]:.1f}  hidden {a[1]:.1f}  swapped {a[2]:.1f} mm;  given-vs-swapped {a[3]:.1f}  given-vs-hidden {a[4]:.1f} mm", flush=True)
