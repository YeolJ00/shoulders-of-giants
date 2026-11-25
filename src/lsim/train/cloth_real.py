"""Fine-tuning the cloth model on real recordings scored by what the cameras saw (claim 3; lsim.real.pgnd).

A real window has no per-vertex truth after its first few frames (PGND's carried points lag the cloth), only observed
point clouds at a few frames. So the loss is the rollout's own: from frame t0 of the window (the fitted, tracked
history), the model runs alone with the held patches following the robot, and each observed cloud's points are scored
by their distance to the predicted surface -- the closest point on the mesh found without gradient, its barycentric
weights then fixed, so the gradient moves the triangle under each point toward it (one-sided, as the evaluation: a
fold's hidden layer is not penalised). Gradients run back through every frame (each frame's forward recomputed in the
backward pass to keep memory to the positions).
"""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from lsim.model.cloth_net import cloth_step, decode, kin_next
from lsim.train.cloth_train import DT, contact_kw, forces
from lsim.train.cloth_windows import collate_cloth, make_cloth_window


def rollout_grad(model, ep, t0, n_frames, P=8, dev="cuda", hidden=("density", "stretch", "bend", "mu"), trunc=0,
                 targets=None, detach_every=0, rng=None, copies=1):
    """As lsim.train.cloth_train.rollout, with gradients: the cloth's predicted positions (n_frames, Vc, 3), frames
    t0 + 1 .. t0 + n_frames, as a tensor on dev. trunc > 0: only the last trunc frames carry gradients (the rollout is
    chaotic -- a 1e-7 m change in the history moves some vertex 0.06 mm one frame on -- and so are its gradients: through
    53 frames their norm reached 1e12-1e14, through 10 frames 0.7-3); with targets (frame indices into the result), the
    trunc frames up to each target. detach_every > 0: the state is cut from the graph every that many frames instead
    (truncated backpropagation; every frame keeps its gradient). ep and t0 may be lists (one rollout per pair, batched):
    then returns (positions (n_frames, N, 3) of every node, the batch) -- score them where batch["is_cloth"] and not
    held. copies > 1: the batch holds that many identical copies of the windows, one after another (a generative
    model's independent samples from one start)."""
    many = isinstance(ep, (list, tuple))
    eps, t0s = (ep, t0) if many else ([ep], [t0])
    rng = np.random.default_rng(0) if rng is None else rng
    wins = [make_cloth_window(e, t + n_frames - 1, P + n_frames - 1, 1, rng) for e, t in zip(eps, t0s)] * copies
    batch = collate_cloth(wins, dev, rng, p_hidden=0.0, augment=False, all_past=True, hidden=hidden)
    win = wins[0]
    Vc = win["n_cloth"]
    true = batch["pos"]
    free = batch["is_cloth"][:, None] & ~batch["held"]
    xs = [true[:, j] for j in range(P)]                                  # the given history, then the model's frames
    carried = None
    out = []

    def frame(xw, pv, lo, n, carried_in):
        bj = dict(batch, pos=xw, vel=(xw - pv) / DT, absent=batch["absent"][:, lo:n], f_ext=batch["f_ext"][:, lo:n],
                  f_state=batch["f_state"][:, lo:n], held=batch["held"][:, lo:n], tau=torch.zeros(P, device=dev),
                  frame_given=torch.ones(P, device=dev), n_past=P)
        bj["kin_next"] = kin_next(bj, true[:, lo:n + 1])
        fi, fc, pr, _ = model(bj)
        c = pr if carried_in is None else carried_in                     # slots from the first window, kept
        w, mu = decode(c, batch)
        F = forces(bj, fi, fc, w)
        kw = contact_kw(model, bj, fc, w)
        kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, -1]
        if kw.get("restore") is not None and kw["restore"].get("st_roll"):   # the shape restoration straight-through
            kw["restore"] = dict(kw["restore"], st=True)                       # over long rollouts (cloth_net)
        return cloth_step(batch, xw[:, -2], xw[:, -1], F[:, -1], w, ~free[:, n - 1], true[:, n], mu, **kw), c

    for k in range(n_frames):
        n = P + k
        lo = n - P
        if detach_every and k and k % detach_every == 0:
            xs = [x.detach() for x in xs]
        xw = torch.stack(xs[lo:n], 1)
        pv = torch.stack(xs[lo - 1:n - 1], 1) if lo > 0 else torch.cat([xw[:, :1] - batch["vel"][:, :1] * DT, xw[:, :-1]], 1)
        tg = [n_frames - 1] if targets is None else targets
        if trunc and not any(0 <= t - k < trunc for t in tg):
            with torch.no_grad():
                x, c = frame(xw, pv, lo, n, carried)
            carried = c if carried is None else carried
        elif carried is None:
            x, carried = frame(xw, pv, lo, n, None)
        else:
            x = checkpoint(lambda a, b, lo=lo, n=n: frame(a, b, lo, n, carried)[0], xw, pv, use_reentrant=False)
        xs.append(x)
        out.append(x)
    X = torch.stack(out, 0)
    return (X, batch) if many else X[:, :Vc]


def cloud_loss(X, cloud, tris, clip=0.05):
    """Mean distance (m) from an observed cloud (M, 3) to the surface X (V, 3, a tensor), each point's closest point on
    the mesh fixed without gradient; points farther than clip (outliers, parts the model has no business near) count
    at clip."""
    from lsim.real.pgnd import project_to_mesh
    t, w, _ = project_to_mesh(cloud.astype(np.float64), X.detach().cpu().numpy().astype(np.float64), tris)
    t = torch.as_tensor(t, device=X.device)
    w = torch.as_tensor(w, device=X.device, dtype=X.dtype)
    q = (X[t] * w[..., None]).sum(1)
    d = (q - torch.as_tensor(cloud, device=X.device, dtype=X.dtype)).norm(dim=-1)
    return d.clamp(max=clip).mean(), d.detach()


def mesh_to_cloud_loss(X, cloud, clip=None):
    """Mean distance (m) from each vertex of X (V, 3, a tensor) to its nearest observed point (found without gradient),
    unclipped by default. cloud_loss alone was gamed: fine-tuned on it, the model blew the cloth up to 17,500x its area
    (a surface everywhere is near every point: 6 mm by that score, 226 mm back to the cloud). Clipped at 5 cm, this term
    did not stop it (cloth_r2 / r3: area up to 350,000x) -- a vertex that has flown off gets no gradient, and flying
    off is the failure it is there to catch. A fold's hidden layer costs its distance to the visible one (a few cm at
    most), so it is down-weighted against cloud_loss by the caller."""
    from scipy.spatial import cKDTree
    _, nn = cKDTree(cloud).query(X.detach().cpu().numpy())
    d = (X - torch.as_tensor(cloud[nn], device=X.device, dtype=X.dtype)).norm(dim=-1)
    return (d if clip is None else d.clamp(max=clip)).mean(), d.detach()


def strain_loss(X, rest, tris, edges=None):
    """Mean |edge strain| of the predicted frames X (T, V, 3) against the rest mesh (V, 3, numpy), as a fraction --
    frames rolled out without gradient add nothing. The real cloth barely stretches (the fitted teacher 0.12% median);
    the model's real rollouts reach 8-47% (v3) and the fine-tunes' blow-ups 10^5 %. edges (E, 2): the mesh edges
    themselves (a rope's chain, a soft body's tet edges), in place of those of tris."""
    if edges is not None:
        E = np.asarray(edges)
    else:
        E = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    E = torch.as_tensor(E, device=X.device)
    L0 = torch.as_tensor(np.linalg.norm(rest[E[:, 0].cpu()] - rest[E[:, 1].cpu()], axis=1), device=X.device, dtype=X.dtype)
    return ((X[:, E[:, 0]] - X[:, E[:, 1]]).norm(dim=-1) / L0 - 1).abs().mean()


def rollout_loss(model, episodes, rng, n_frames, dev="cuda", n=2, detach_every=10, P=8, p_given=0.0, gen=""):
    """Long-rollout loss on teacher episodes: n episodes, each rolled out alone n_frames from a random start (true
    history, all slots hidden), the mean distance (cm) of every free cloth vertex from the teacher's at every frame
    (distance, not squared: errors reach centimetres by a few seconds), the graph cut every detach_every frames. The
    one-step loss sees only states a frame off the teacher's; this trains on the states the model's own rollouts reach
    (v3: 8-67 mm from its teacher after 1 s on PGND's scenes). p_given: the share of steps with the four material
    slots given instead of hidden (all hidden, the model learned to ignore a given material: on PGND given and hidden
    score within 0.3 mm, where the teacher gains 5 mm from its fitted material). gen == "energy" (a generative
    model): each episode rolled out twice from the same start (one batch, independent noise), the energy score
    (|X1 - y| + |X2 - y| - |X1 - X2|) / 2 per free vertex and frame in place of the distance."""
    eps = [episodes[i] for i in rng.integers(len(episodes), size=4 * n)]
    eps = [e for e in eps if e.n_frames > P + n_frames + 1][:n]
    t0s = [int(rng.integers(P - 1, e.n_frames - n_frames - 1)) for e in eps]
    hidden = () if rng.random() < p_given else ("density", "stretch", "bend", "mu")
    m = len(eps)
    if gen not in ("", "energy"):
        raise ValueError(f"gen {gen}: only 'energy'")
    X, batch = rollout_grad(model, eps, t0s, n_frames, P=P, dev=dev, detach_every=detach_every, rng=rng, hidden=hidden,
                            copies=2 if gen == "energy" else 1)              # energy: each start twice, two samples
    true = batch["pos"][:, P:P + n_frames].transpose(0, 1)               # (n_frames, N, 3)
    free = (batch["is_cloth"][:, None] & ~batch["held"])[:, P:P + n_frames].transpose(0, 1)
    if gen == "energy":                                                  # the second copy's nodes follow the first's
        N1 = int((batch["sample_of"] < m).sum())
        assert 2 * N1 == X.shape[1]
        X1, X2, true, free = X[:, :N1], X[:, N1:], true[:, :N1], free[:, :N1]
        d1, d2 = (X1 - true).norm(dim=-1)[free], (X2 - true).norm(dim=-1)[free]
        d12 = ((X1 - X2) ** 2).sum(-1)[free].add(1e-12).sqrt()
        es = 0.5 * (d1.mean() + d2.mean()) - 0.5 * d12.mean()
        return es / 1e-2, {"roll_mm": 1e3 * d1.mean().item(), "roll_spread_mm": 1e3 * d12.mean().item(),
                           "roll_end_mm": 1e3 * (X1[-1] - true[-1]).norm(dim=-1)[free[-1]].median().item()}
    d = (X - true).norm(dim=-1)[free]
    return d.mean() / 1e-2, {"roll_mm": 1e3 * d.mean().item(), "roll_end_mm": 1e3 * (X[-1] - true[-1]).norm(dim=-1)[free[-1]].median().item()}
