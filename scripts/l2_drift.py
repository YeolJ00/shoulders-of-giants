"""L2 -- resting drift diagnostic (DIRECTION.md "Resting objects drifted").

    $L scripts/l2_drift.py <ckpt> [n_episodes=12] [modes=free,tf] [train_data=data/datasets/v0]

30-frame rollouts from frames 30 and 90 of the first n episodes of the training data's shard 0
(v0 unless given; v0r for its ramps) and of v0_test (held-out objects); bodies the teacher kept at rest and moving bodies reported
apart (median and p90 of the mean VERTEX error, so rotation counts: the body-origin error hid
~1.4 deg per step of spurious rotation at rest in every trained variant). Modes: free (the rollout), tf (one step from the
true history at every frame: the per-step bias), anch<tau> (the training layout at inference:
8 given frames + k <= 7 flagged chunk frames at noise level tau, re-anchored every 8 frames)."""
import copy, glob, sys
import numpy as np, torch
from lsim.model.net import build_model
from lsim.model.rollout import rollout
from lsim.train.train import collate, damping_slot, physics_step, slots
from lsim.train.windows import F_EMPTY, F_GIVEN, PARAMS, load_shard, make_window
from lsim.data.episode import rotate


def vert_err(b, p, t):
    """Mean vertex distance between two poses (..., 7) of body b: position AND orientation errors."""
    v = b.verts.astype(float)
    return np.linalg.norm(rotate(p[3:], v) + p[:3] - rotate(t[3:], v) - t[:3], axis=1).mean()


@torch.no_grad()
def rollout_anchored(model, ep, t0, H, tau_inf, P=8, dev="cuda"):
    rng = np.random.default_rng(0)
    view = copy.copy(ep); view.q = ep.q.copy()
    ids = [b.body for b in ep.bodies]
    dyn = np.array([b.role == "dynamic" for b in ep.bodies])
    carried = None
    for t in range(t0, t0 + H):
        a = t0 + P * ((t - t0) // P)
        k = t - a
        w = make_window(view, a, P, k, rng, p_hide_grip=0.0)
        b = collate([w], torch.device(dev), rng, p_hist_noise=0.0, augment=False, all_past=True, hidden=())
        Fw = b["pos"].shape[1]
        tau = torch.cat([torch.zeros(P, device=dev), torch.full((k,), float(tau_inf), device=dev)])
        fg = torch.cat([torch.ones(P, device=dev), torch.zeros(k, device=dev)])
        f_contact, f_ext_pred, pred = model(dict(b, tau=tau, frame_given=fg),
                                            lambda pr: slots(b, pr if carried is None else carried)[0])
        if carried is None:
            carried = pred
        m, c, J = slots(b, carried)
        j = P - 1 + k
        st = b["f_state"][:, j]
        f_ext = torch.where((st == F_GIVEN)[:, None], b["f_ext"][:, j], torch.zeros_like(b["f_ext"][:, j]))
        f_ext = torch.where((st == F_EMPTY)[:, None], f_ext_pred[:, j], f_ext)
        Q = b["body_q_win"]
        new = physics_step(b, Q[:, j + 1], Q[:, j], f_contact[:, j] + f_ext, b["pos"][:, j], m, c, J,
                           damping_slot(b, carried)).cpu().numpy()
        for i, bi in enumerate(ids):
            if dyn[i]:
                view.q[t + 1, bi] = new[i]
    return view.q[t0 + 1:t0 + H + 1][:, ids], ep.q[t0 + 1:t0 + H + 1][:, ids]


def main():
    ckpt, n_eps = sys.argv[1], int(sys.argv[2])
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["free", "tf", "anch0.1", "anch0.3"]
    dev = torch.device("cuda")
    ck = torch.load(ckpt, map_location=dev)
    model = build_model(ck.get("args", {}), len(PARAMS)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    def eps_of(pattern, first_only):
        shards = sorted(glob.glob(pattern))[:1] if first_only else sorted(glob.glob(pattern))
        return [e for s in shards for e in load_shard(s[:-4]) if any(b.role == "dynamic" and b.type == "rigid" for b in e.bodies)]

    train = sys.argv[4] if len(sys.argv) > 4 else "data/datasets/v0"
    sets = {"train": eps_of(f"{train}/shard_*.npz", True)[:100], "test": eps_of("data/datasets/v0_test/shard_*.npz", False)}
    for name, eps in sets.items():
        for mode in modes:
            rest_d, move_e = [], []
            for e in eps[:n_eps]:
                for t0 in (30, 90):
                    if mode == "free":
                        pred, true, _ = rollout(model, e, t0, 30)
                    elif mode == "tf":
                        pred = np.stack([rollout(model, e, t, 1)[0][0] for t in range(t0, t0 + 30)])
                        true = e.q[t0 + 1:t0 + 31][:, [b.body for b in e.bodies]]
                    else:
                        pred, true = rollout_anchored(model, e, t0, 30, float(mode[4:]))
                    for i, b in enumerate(e.bodies):
                        if b.role != "dynamic" or b.type != "rigid":
                            continue
                        rest = (np.linalg.norm(true[-1, i, :3] - true[0, i, :3]) < 1e-3
                                and np.linalg.norm(true[0, i, :3] - e.q[t0, b.body, :3]) < 1e-3)
                        if mode == "tf":                                   # mean one-step error
                            d = np.mean([vert_err(b, pred[h, i], true[h, i]) for h in range(len(pred))])
                        else:                                              # error at frame 30
                            d = vert_err(b, pred[-1, i], true[-1, i])
                        (rest_d if rest else move_e).append(1000 * d)
            what = "mean one-step error" if mode == "tf" else "error at 30 frames"
            print(f"[{name:5s} {mode:8s}] {what}: resting n {len(rest_d):3d} median {np.median(rest_d):7.2f} p90 {np.percentile(rest_d, 90):7.2f} mm"
                  f" | moving n {len(move_e):3d} median {np.median(move_e):7.2f} p90 {np.percentile(move_e, 90):7.2f} mm", flush=True)


if __name__ == "__main__":
    main()
