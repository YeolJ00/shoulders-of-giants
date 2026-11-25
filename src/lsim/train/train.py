"""Training v0 (rigid objects and hinged parts; the gripper follows its recorded path).

Windows of P past + K chunk frames; 0-7 past frames present before "now" (DIRECTION.md "Time").
Each chunk frame gets its own noise level tau (logit-normal, as in diffusion forcing), applied
as rigid random-walk noise to the dynamic bodies' poses; every chunk step is one physics step
from the noised state of the frame before it, scored against the true next frame (step_loss).
Inference rolls out frame by frame. The original design -- the network denoising the whole
chunk while the physics layer integrates it open loop (step_loss_chunk) -- is kept as an
ablation; see its docstring for why it lost.

Slots (DIRECTION.md "Slots", PHASE1_AMENDMENT.md): parameter groups (mass, friction,
restitution, centre of mass, inertia scale, mass-model flag, joint damping) are hidden with
p = 0.5 each (10% of windows all hidden, 10% none). The network predicts every slot; the exact
physics layer integrates each dynamic rigid body from its state at "now" with the GIVEN mass,
centre of mass and inertia where those slots are given and the PREDICTED ones where hidden, so
a hidden mass must be inferred to move the object right. External forces: given gripper forces
enter the physics layer exactly; empty force slots (unseen pokes, the gripper's force when
hidden) are predicted by the network. Predicted slots reach the physics layer detached: the
parameter head is trained by the slot loss alone.

Loss: one-step vertex position error over the chunk frames (cm^2), plus the hidden parameter
slots and the hidden external forces.

    $L -m lsim.train.train --data data/datasets/v0 --steps 20000
"""
import argparse
import glob
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

from lsim.model.net import Simulator, build_model
from lsim.model.physics import joint_pose, joint_step, qexp, qmul, qrot, rigid_step, wrench
from lsim.train.windows import CENTRE, F_EMPTY, F_GIVEN, PARAMS, SCALE, load_shard, make_window

DT = 1.0 / 30.0
GROUPS = [[0], [1], [2], [3, 4, 5], [6, 7, 8], [9], [10]]          # hidden together
GROUP_NAMES = ("mass", "mu", "restitution", "com", "inertia", "solid", "joint_damping")
SOLID = PARAMS.index("solid")
FORCE_UNIT = 0.1                                                    # N: asinh scale of force targets
SIGMA = 0.01                # m per frame: displacement noise scale (fixed: the sampler cannot see the chunk)


def _yaw(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)


def collate(wins, device, rng, p_hidden=0.5, p_hist_noise=0.2, augment=True, all_past=False, hidden=None):
    """Concatenate windows (vertices, bodies with per-window offsets), draw which parameter slots
    are hidden, which past frames are present, and a random yaw rotation + translation per window
    (an approximate symmetry about gravity)."""
    keys = ("pos", "vel", "rest", "f_ext", "f_state", "body_of", "role", "type", "sample_of", "edges", "absent",
            "params", "param_has", "param_given", "body_type", "body_role", "body_q_now", "body_q_prev",
            "body_mass", "body_com", "body_inertia", "body_size", "body_lam", "body_R", "body_q_chunk", "body_q_win",
            "vert_local", "body_joint", "joint_axis", "joint_pivot", "joint_lim", "surface", "normal")
    B = {k: [] for k in keys}
    nv = nb = 0
    P = wins[0]["n_past"]
    for s, w in enumerate(wins):
        n, nbody, Fw = w["pos"].shape[1], len(w["body_role"]), w["pos"].shape[0]
        R = _yaw(rng.uniform(-np.pi, np.pi)) if augment else np.eye(3, dtype=np.float32)
        t = np.r_[rng.uniform(-0.2, 0.2, 2), 0.0].astype(np.float32) if augment else np.zeros(3, np.float32)
        th = np.arctan2(R[1, 0], R[0, 0]); qz = np.array([0, 0, np.sin(th / 2), np.cos(th / 2)], np.float32)
        B["pos"].append(torch.from_numpy(w["pos"] @ R.T + t).transpose(0, 1))
        B["vel"].append(torch.from_numpy(w["vel"] @ R.T).transpose(0, 1))
        B["f_ext"].append(torch.from_numpy(w["f_ext"] @ R.T).transpose(0, 1))
        B["f_state"].append(torch.from_numpy(w["f_state"]).transpose(0, 1))
        B["rest"].append(torch.from_numpy(w["rest"]))
        B["surface"].append(torch.from_numpy(w["surface"]))
        B["normal"].append(torch.from_numpy(w["normal"] @ R.T))            # static surfaces' outward normals
        B["body_of"].append(torch.from_numpy(w["body_of"]) + nb)
        B["role"].append(torch.from_numpy(w["role"]))
        B["type"].append(torch.from_numpy(w["type"]))
        B["sample_of"].append(torch.full((n,), s))
        B["edges"].append(torch.from_numpy(w["edges"]).long().reshape(-1, 2) + nv)
        # present past frames: 0-7 before "now"; frames before the episode start are absent too
        n_prev = P - 1 if all_past else int(rng.integers(0, P))
        absent = np.zeros(Fw, bool); absent[:P - 1 - n_prev] = True; absent |= w["frames"] < 0
        B["absent"].append(torch.from_numpy(np.broadcast_to(absent, (n, Fw)).copy()))
        # parameter slots: groups hidden with p_hidden; 10% of windows all hidden, 10% none
        hide = rng.random((nbody + 1, len(GROUPS))) < p_hidden
        u = rng.random()
        hide[:] = True if (u < 0.1 and p_hidden > 0) else (False if u < 0.2 or p_hidden == 0 else hide)
        if hidden is not None:                                            # evaluation: exactly these groups hidden
            hide[:] = np.isin(GROUP_NAMES, list(hidden))[None]
        given = np.ones((nbody + 1, len(PARAMS)), bool)
        for g, idx in enumerate(GROUPS):
            given[:, idx] = ~hide[:, g:g + 1]
        has = np.concatenate([w["param_has"], np.zeros((1, len(PARAMS)), bool)])     # + static pseudo-body
        B["params"].append(torch.from_numpy(np.concatenate([w["params"], np.zeros((1, len(PARAMS)), np.float32)])))
        B["param_has"].append(torch.from_numpy(has))
        B["param_given"].append(torch.from_numpy(given & has))
        def turn(qa):                                                   # poses (..., 7) under the yaw + shift
            qa = qa.copy()
            qa[..., :3] = qa[..., :3] @ R.T + t
            qa[..., 3:] = qmul(torch.from_numpy(np.broadcast_to(qz, qa.shape[:-1] + (4,)).copy()), torch.from_numpy(qa[..., 3:])).numpy()
            return qa
        qn, qp = turn(w["body_q_now"]), turn(w["body_q_prev"])
        qc = turn(w["body_q_chunk"])                                      # (K, nbody, 7)
        pad = {"body_type": np.zeros(1, int), "body_role": np.zeros(1, int), "body_q_now": np.array([[0, 0, 0, 0, 0, 0, 1.0]]),
               "body_q_prev": np.array([[0, 0, 0, 0, 0, 0, 1.0]]), "body_mass": np.ones(1), "body_com": np.zeros((1, 3)),
               "body_inertia": np.eye(3)[None], "body_size": np.ones(1), "body_lam": np.ones((1, 3)), "body_R": np.eye(3)[None],
               "body_joint": np.zeros(1, int), "joint_axis": np.zeros((1, 3)), "joint_pivot": np.zeros((1, 3)),
               "joint_lim": np.zeros((1, 2))}
        vals = dict(w, body_q_now=qn, body_q_prev=qp, joint_axis=w["joint_axis"] @ R.T, joint_pivot=w["joint_pivot"] @ R.T + t)
        for k, v in pad.items():
            B[k].append(torch.from_numpy(np.concatenate([np.asarray(vals[k]), v.astype(np.asarray(vals[k]).dtype)])))
        B["body_q_chunk"].append(torch.from_numpy(np.concatenate([qc, np.broadcast_to(pad["body_q_now"], (qc.shape[0], 1, 7))], 1)).transpose(0, 1))
        qw = turn(w["body_q_win"])                                        # (Fw + 1, nbody, 7), from the frame before
        B["body_q_win"].append(torch.from_numpy(np.concatenate([qw, np.broadcast_to(pad["body_q_now"], (qw.shape[0], 1, 7))], 1)).transpose(0, 1))
        vl = [torch.from_numpy(v) for v in w["body_verts"]] + [torch.zeros(w["n_static"], 3)]
        B["vert_local"].append(torch.cat(vl))
        nv += n
        nb += nbody + 1
    out = {k: torch.cat(v).to(device) for k, v in B.items()}
    for k in ("pos", "vel", "f_ext", "rest", "body_q_now", "body_q_prev", "body_mass", "body_com", "body_inertia",
              "body_size", "body_lam", "body_R", "body_q_chunk", "body_q_win", "vert_local", "params", "joint_axis",
              "joint_pivot", "joint_lim", "normal"):
        out[k] = out[k].float()
    out["n_bodies"] = nb
    out["n_past"] = P
    out["p_hist_noise"] = p_hist_noise
    return out


def slots(batch, pred):
    """Mass, centre of mass (body frame) and inertia per unit mass for the physics layer: the true
    values where the slot is given, decoded predictions where hidden (dynamic rigid bodies only)."""
    rigid = (batch["body_role"] == 2) & (batch["body_type"] == 0)
    given = batch["param_given"]
    ce, sc = torch.as_tensor(CENTRE, device=pred.device), torch.as_tensor(SCALE, device=pred.device)
    raw = pred * sc + ce
    m_true = batch["body_mass"]
    m = torch.where(rigid & ~given[:, 0], torch.exp(raw[:, 0].clamp(-8, 3)), m_true)
    c = torch.where((rigid & ~given[:, 3])[:, None], raw[:, 3:6] * batch["body_size"][:, None], batch["body_com"])
    s = torch.exp(raw[:, 6:9].clamp(-2, 2)) * batch["body_lam"]
    J_pred = batch["body_R"] @ torch.diag_embed(s) @ batch["body_R"].transpose(-1, -2)
    J_true = batch["body_inertia"] / m_true[:, None, None].clamp_min(1e-6)
    J = torch.where((rigid & ~given[:, 6])[:, None, None], J_pred, J_true)
    return m, c, J


def frame_velocity(q_prev, q_now, c):
    """Frame-average linear velocity of the point c (body frame) and angular velocity, from two
    poses one frame apart."""
    v = (q_now[:, :3] + qrot(q_now[:, 3:], c) - q_prev[:, :3] - qrot(q_prev[:, 3:], c)) / DT
    inv = q_prev[:, 3:] * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=c.device)
    dq = qmul(q_now[:, 3:], inv)
    dq = dq * torch.where(dq[:, 3:] < 0, -1.0, 1.0)                    # shortest rotation
    s = torch.linalg.norm(dq[:, :3], dim=-1, keepdim=True)
    w = 2.0 * torch.atan2(s, dq[:, 3:]) * dq[:, :3] / s.clamp_min(1e-12) / DT
    return v, w


def integrate(batch, f_tot, K, P, m, c, J):
    """Integrate every dynamic rigid body (role 2, type 0) through the chunk from its state at
    "now"; return predicted world positions (N, K, 3) of all vertices (non-integrated bodies keep
    their given positions)."""
    pos, body = batch["pos"], batch["body_of"]
    dyn = (batch["body_role"] == 2) & (batch["body_type"] == 0)
    qn = batch["body_q_now"]
    q = qn[:, 3:]
    p = qn[:, :3] + qrot(q, c)                                    # centre of mass now
    v, w = frame_velocity(batch["body_q_prev"], qn, c)
    mass, I = m.clamp_min(1e-4), m[:, None, None] * J
    out = pos[:, P:P + K].clone()
    cur = pos[:, P - 1]
    for k in range(K):
        F, T = wrench(f_tot[:, P - 1 + k], cur, body, p, batch["n_bodies"])
        p1, q1, v1, w1 = rigid_step(p, q, v, w, F, T, mass, I, DT)
        msk = dyn[:, None]
        p, q, v, w = torch.where(msk, p1, p), torch.where(msk, q1, q), torch.where(msk, v1, v), torch.where(msk, w1, w)
        xk = p[body] + qrot(q[body], batch["vert_local"] - c[body])
        dv = dyn[body]
        out[dv, k] = xk[dv]
        cur = torch.where(dv[:, None], xk, pos[:, P + k])
    return out


def damping_slot(batch, pred):
    """Joint damping of hinged parts: given where the slot is given, else the prediction."""
    i = PARAMS.index("joint_damping")
    v = torch.where(batch["param_given"][:, i], batch["params"][:, i], pred[:, i])
    return torch.exp(v * float(SCALE[i]) + float(CENTRE[i]))


def physics_step(batch, cur, prv, f, x, m, c, J, damping):
    """One frame for every body from two origin poses (nb, 7): dynamic rigid bodies by
    rigid_step about their centre of mass, hinged parts along their joint, everything else
    unchanged. f, x: per-vertex forces and positions at `cur`. Returns new origin poses."""
    body, nb = batch["body_of"], batch["n_bodies"]
    rigid = (batch["body_role"] == 2) & (batch["body_type"] == 0)
    hinge = batch["body_joint"] > 0
    p = cur[:, :3] + qrot(cur[:, 3:], c)
    v, w = frame_velocity(prv, cur, c)
    F, T = wrench(f, x, body, p, nb)
    p1, q1, _, _ = rigid_step(p, cur[:, 3:], v, w, F, T, m.clamp_min(1e-4), m[:, None, None] * J, DT)
    new_r = torch.cat([p1 - qrot(q1, c), q1], -1)
    Fh, Th = wrench(f, x, body, batch["joint_pivot"], nb)
    new_h = joint_step(batch["body_joint"], batch["joint_axis"], batch["joint_pivot"], batch["joint_lim"], prv, cur,
                       Fh, Th, m.clamp_min(1e-4), c, m[:, None, None] * J, damping, DT)
    return torch.where(rigid[:, None], new_r, torch.where(hinge[:, None], new_h, cur))


def step_loss_chunk(model, batch, rng, K, dev, tau_fixed=None):
    """Ablation (the original design): diffusion forcing over the chunk -- the network sees the
    noised true chunk, the physics layer integrates its forces open loop from "now", x-prediction
    loss. tau_fixed sets every chunk frame's noise level (1.0 = predict from the past alone).
    Overfitting one episode it ended 26 mm off after 8 frames (sampled): the network never sees
    the physics layer's own state, so force errors accumulate unseen, and at high noise there is
    no geometry to find contacts on."""
    P = batch["n_past"]
    pos = batch["pos"]
    dyn_v = (batch["role"] == 2) & (batch["type"] == 0)                # integrated vertices
    Fw = pos.shape[1]
    # diffusion forcing on the chunk; light noise on the history of some windows
    tau = torch.sigmoid(torch.randn(K, device=dev)) if tau_fixed is None else torch.full((K,), float(tau_fixed), device=dev)
    d = torch.diff(pos, dim=1)                                             # (N, Fw-1, 3): frame j -> j+1
    scale = SIGMA
    true = pos[:, P:P + K]
    dc = d[:, P - 1:]
    noisy = pos[:, P - 1:P] + torch.cumsum((1 - tau[None, :, None]) * dc + tau[None, :, None] * torch.randn_like(dc) * scale, 1)
    pos_in = pos.clone()
    pos_in[dyn_v, P:P + K] = noisy[dyn_v]
    tau_all = torch.cat([torch.zeros(P, device=dev), tau])
    if tau_fixed is None and rng.random() < batch["p_hist_noise"]:
        th = float(rng.uniform(0.0, 0.1))
        dh = d[:, :P - 1]
        dh = (1 - th) * dh + th * torch.randn_like(dh) * scale
        hist = pos[:, P - 1:P] - torch.flip(torch.cumsum(torch.flip(dh, [1]), 1), [1])   # anchored at "now"
        pos_in[dyn_v, :P - 1] = hist[dyn_v]
        tau_all[:P - 1] = th
    prev = torch.cat([pos_in[:, :1] - batch["vel"][:, :1] * DT, pos_in[:, :-1]], 1)
    vel_in = (pos_in - prev) / DT
    batch_in = dict(batch, pos=pos_in, vel=vel_in, tau=tau_all,
                    frame_given=torch.cat([torch.ones(P, device=dev), torch.zeros(K, device=dev)]))

    # the physics layer uses the predicted slots but does not train them: each output is trained
    # by its own loss (with the gradient, the position loss used a hidden mass as a gain knob and
    # drove the mass error from 0.9 to 4.4 log units in five steps)
    def slot_mass(pred):
        return slots(batch, pred.detach())[0]
    f_contact, f_ext_pred, pred = model(batch_in, slot_mass)
    m, c, J = slots(batch, pred.detach())
    state = batch["f_state"]
    f_ext = torch.where((state == F_GIVEN)[..., None], batch["f_ext"], torch.zeros_like(batch["f_ext"]))
    f_ext = torch.where((state == F_EMPTY)[..., None], f_ext_pred, f_ext)
    out = integrate(batch, f_contact + f_ext, K, P, m, c, J)
    wgt = 1.0 / tau.clamp_min(0.05) ** 2
    err = ((out - true) ** 2).sum(-1)[dyn_v]                               # (Nd, K) m^2
    l_pos = (err * wgt[None]).mean() / wgt.mean() / 1e-4                    # cm^2
    # hidden parameter slots (x-prediction at tau = 1): MSE in slot units, BCE for the flag
    hid = batch["param_has"] & ~batch["param_given"]
    cont = hid.clone(); cont[:, SOLID] = False
    l_par = ((pred - batch["params"]) ** 2)[cont].mean() if cont.any() else pred.sum() * 0
    if hid[:, SOLID].any():
        l_par = l_par + Fn.binary_cross_entropy_with_logits(pred[hid[:, SOLID], SOLID], batch["params"][hid[:, SOLID], SOLID])
    # hidden external forces, compressed (pokes are a few newtons, the gripper up to 60 N)
    emp = (state == F_EMPTY)[:, :Fw - 1]
    l_f = (Fn.smooth_l1_loss(torch.asinh(f_ext_pred[:, :Fw - 1][emp] / FORCE_UNIT),
                             torch.asinh(batch["f_ext"][:, :Fw - 1][emp] / FORCE_UNIT)) if emp.any() else pred.sum() * 0)
    loss = l_pos + l_par + l_f
    with torch.no_grad():
        mass_err = (pred[:, 0] - batch["params"][:, 0])[hid[:, 0]].abs().mean() * SCALE[0] if hid[:, 0].any() else torch.tensor(float("nan"))
    return loss, {"pos_rmse_mm": 1000 * err.mean().sqrt().item(), "rmse_first_mm": 1000 * err[:, 0].mean().sqrt().item(),
                  "rmse_last_mm": 1000 * err[:, -1].mean().sqrt().item(), "l_pos": l_pos.item(), "l_par": l_par.item(),
                  "l_f": l_f.item(), "mass_abs_log": mass_err.item()}


def step_loss_ar(model, batch, rng, K, dev):
    """Ablation and evaluation: no noise; the chunk is rolled out frame by frame, the network
    re-run at every step on the physics layer's own positions, seeing only frames up to the
    current one (K network passes per window, the k-th over P + k frames). As a training loss it
    was ~5x the cost of step_loss and behind it at equal steps (55 vs 19 mm at 250 steps)."""
    P = batch["n_past"]
    pos, body = batch["pos"], batch["body_of"]
    dyn_v = batch["role"] == 2                                           # rigid objects and hinged parts
    state = batch["f_state"]
    pos_in = pos.clone()
    out, preds, fexts = [], [], []
    prv, cur_q = batch["body_q_prev"], batch["body_q_now"]
    for k in range(K):
        n = P + k                                                        # frames up to the current one
        x = pos_in[:, :n]
        prev = torch.cat([x[:, :1] - batch["vel"][:, :1] * DT, x[:, :-1]], 1)
        b_in = dict(batch, pos=x, vel=(x - prev) / DT, absent=batch["absent"][:, :n], f_ext=batch["f_ext"][:, :n],
                    f_state=state[:, :n], tau=torch.zeros(n, device=dev),
                    frame_given=torch.cat([torch.ones(P, device=dev), torch.zeros(k, device=dev)]))
        if k == 0:
            f_contact, f_ext_pred, pred = model(b_in, lambda pr: slots(batch, pr.detach())[0])
            m, c, J = slots(batch, pred.detach())
            damping = damping_slot(batch, pred.detach())
        else:
            f_contact, f_ext_pred, _ = model(b_in, lambda pr: m)
        j = P - 1 + k
        f_ext = torch.where((state[:, j] == F_GIVEN)[:, None], batch["f_ext"][:, j], torch.zeros_like(batch["f_ext"][:, j]))
        f_ext = torch.where((state[:, j] == F_EMPTY)[:, None], f_ext_pred[:, j], f_ext)
        fexts.append(f_ext_pred[:, j])
        if getattr(getattr(model, "module", model), "direct", False):   # B1/B2: vertex step, rigid per body
            xk = rigid_project(2 * pos_in[:, j] - pos_in[:, j - 1] + f_contact[:, j] * DT * DT, batch["vert_local"], body,
                               batch["n_bodies"], dyn_v)
        else:
            new = physics_step(batch, cur_q, prv, f_contact[:, j] + f_ext, pos_in[:, j], m, c, J, damping)
            prv, cur_q = cur_q, new
            xk = new[body, :3] + qrot(new[body, 3:], batch["vert_local"])
        cur = torch.where(dyn_v[:, None], xk, pos[:, P + k])
        pos_in = pos_in.clone()
        pos_in[:, P + k] = cur
        out.append(cur)
    out = torch.stack(out, 1)
    true = pos[:, P:P + K]
    err = ((out - true) ** 2).sum(-1)[dyn_v]
    l_pos = err.mean() / 1e-4
    hid = batch["param_has"] & ~batch["param_given"]
    cont = hid.clone(); cont[:, SOLID] = False
    l_par = ((pred - batch["params"]) ** 2)[cont].mean() if cont.any() else pred.sum() * 0
    if hid[:, SOLID].any():
        l_par = l_par + Fn.binary_cross_entropy_with_logits(pred[hid[:, SOLID], SOLID], batch["params"][hid[:, SOLID], SOLID])
    fe = torch.stack(fexts, 1)
    emp = state[:, P - 1:P - 1 + K] == F_EMPTY
    l_f = (Fn.smooth_l1_loss(torch.asinh(fe[emp] / FORCE_UNIT), torch.asinh(batch["f_ext"][:, P - 1:P - 1 + K][emp] / FORCE_UNIT))
           if emp.any() else pred.sum() * 0)
    still = ((pos[:, P + K - 1] - pos[:, P - 1]).norm(dim=-1) < 1e-3)[dyn_v]    # vertices the teacher kept at rest
    return l_pos + l_par + l_f, {"pos_rmse_mm": 1000 * err.mean().sqrt().item(),
                                 "rmse_first_mm": 1000 * err[:, 0].mean().sqrt().item(),
                                 "rmse_last_mm": 1000 * err[:, -1].mean().sqrt().item(), "l_pos": l_pos.item(),
                                 "rest_last_mm": 1000 * err[still, -1].mean().sqrt().item() if still.any() else float("nan"),
                                 "l_par": l_par.item(), "l_f": l_f.item()}


def rigid_project(x, vl, body, nb, mask):
    """Each body's vertices x (N, 3) replaced by the best rigid placement of its body-frame points
    vl (Kabsch, batched over bodies), where mask; B2's rollouts do this after every step."""
    one = torch.ones_like(body, dtype=x.dtype)
    cnt = torch.zeros(nb, device=x.device, dtype=x.dtype).index_add_(0, body, one).clamp_min(1)[:, None]
    cx = torch.zeros(nb, 3, device=x.device, dtype=x.dtype).index_add_(0, body, x) / cnt
    cl = torch.zeros(nb, 3, device=x.device, dtype=x.dtype).index_add_(0, body, vl) / cnt
    H = torch.zeros(nb, 3, 3, device=x.device, dtype=x.dtype).index_add_(
        0, body, (vl - cl[body])[:, :, None] * (x - cx[body])[:, None, :])
    U, _, Vt = torch.linalg.svd(H)
    V, Ut = Vt.transpose(-1, -2), U.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(V @ Ut))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    xp = ((V @ D @ Ut)[body] @ (vl - cl[body])[..., None])[..., 0] + cx[body]
    return torch.where(mask[:, None], xp, x)


def step_loss(model, batch, rng, K, dev, sig_p=0.01, sig_r=0.05, sig_h=(0.0, 0.0), w_sig=0.0, pf=0, l1=False, mix=False):
    """The training step: every chunk step is ONE physics step from the window's own (noised)
    state at the frame before it -- GNS-style targets with noise injection, so contacts are
    found on rigid, nearly physical geometry and the model learns to pull a perturbed state
    back. Dynamic bodies' chunk poses get rigid random-walk noise (sig_p m and sig_r rad per
    frame) scaled by each frame's tau (diffusion forcing's per-frame levels, logit-normal).
    Inference rolls out frame by frame (lsim.model.rollout). Overfitting one episode: 4-9 mm
    after an 8-frame rollout at 1,000-1,250 steps, against 26 mm for step_loss_chunk at 1,500.
    Hinged parts move along their joint (physics_step), with noise along the joint.
    sig_h = (lo, hi) m: the past frames of every dynamic rigid body also get a rigid random walk
    anchored at "now" (zero at frame P-1, per-body step size log-uniform in [lo, hi]) that is
    NOT flagged (tau stays 0): the first step then starts from a true pose with a perturbed
    velocity, and its target is still the true next frame -- the given-frame regime a rollout
    runs in, where the model's own small errors come back as history.
    w_sig > 0 (m): step k is weighted by w_sig^2 / (w_sig^2 + s_k^2), s_k the translational noise
    std of its starting frame, so the clean first step -- the only one a rollout takes -- is not
    drowned by the corrections of centimetre-scale noise.
    pf > 0 (pushforward): the first r chunk frames (r uniform in 1..pf per batch) are replaced by
    the model's own no-grad rollout from "now", flagged as GIVEN frames with tau 0 -- exactly
    what a rollout feeds back -- and the steps from them are scored against the true next
    frames: the model learns to undo its own errors in the regime it runs in (Brandstetter et
    al. 2022). Costs r extra forward passes without gradients.
    l1: the position loss is the mean vertex DISTANCE (cm) instead of its square, so every vertex
    pulls with the same strength however small its error -- a resting body 0.5 mm off weighed
    ~1e-3 of an impact under the squared loss, yet that error is what compounds in a rollout.
    mix: distance (cm) + squared distance (cm^2): linear below ~5 mm, quadratic above, so small
    errors pull like l1 and large ones still dominate as in the squared loss (l1 alone gave
    0.03 mm per step at rest but let moving bodies diverge)."""
    P = batch["n_past"]
    pos, body = batch["pos"], batch["body_of"]
    dyn_v = batch["role"] == 2                                           # rigid objects and hinged parts
    dyn = (batch["body_role"] == 2) & (batch["body_type"] == 0)
    hinge = batch["body_joint"] > 0
    Fw = pos.shape[1]
    Q = batch["body_q_win"]                                              # (nb, Fw + 1, 7); index j + 1 = frame j
    nb = Q.shape[0]
    tau = torch.sigmoid(torch.randn(K, device=dev))
    wp = torch.cumsum(torch.randn(nb, K, 3, device=dev) * sig_p, 1) * tau[None, :, None]
    wr = torch.cumsum(torch.randn(nb, K, 3, device=dev) * sig_r, 1) * tau[None, :, None]
    if sig_h[1] > 0:
        s_b = torch.exp(torch.empty(nb, device=dev).uniform_(float(np.log(sig_h[0])), float(np.log(sig_h[1])))) * dyn
        walk = lambda: torch.flip(torch.cumsum(torch.randn(nb, P, 3, device=dev), 1), [1])   # Q index P - 1 - i: i + 1 steps
        hp = walk() * s_b[:, None, None]
        hr = walk() * (s_b / batch["body_size"].clamp_min(0.02))[:, None, None]
        Q = torch.cat([torch.cat([Q[:, :P, :3] + hp, qmul(qexp(hr, 1.0), Q[:, :P, 3:])], -1), Q[:, P:]], 1)
        Qv = Q[body, 1:P]                                                # frames 0 .. P-2 (P-1 is untouched)
        xh = Qv[..., :3] + qrot(Qv[..., 3:], batch["vert_local"][:, None].expand(-1, P - 1, -1))
        pos = pos.clone()
        pos[:, :P - 1] = torch.where((dyn[body] & dyn_v)[:, None, None], xh, pos[:, :P - 1])
    Qn = Q.clone()
    ch = Q[:, P + 1:P + 1 + K]
    noisy = torch.cat([ch[..., :3] + wp, qmul(qexp(wr, 1.0), ch[..., 3:])], -1)
    jn = torch.where((batch["body_joint"] == 1)[:, None], wp[..., 0], wr[..., 0])   # hinged parts: noise along the joint
    noisy_h = torch.stack([joint_pose(batch["body_joint"], batch["joint_axis"], batch["joint_pivot"], ch[:, k], jn[:, k])
                           for k in range(K)], 1)
    Qn[:, P + 1:P + 1 + K] = torch.where(dyn[:, None, None], noisy, torch.where(hinge[:, None, None], noisy_h, ch))
    pos_in = pos.clone()
    vl = batch["vert_local"]
    for k in range(K):
        qk = Qn[body, P + 1 + k]
        pos_in[dyn_v, P + k] = (qk[:, :3] + qrot(qk[:, 3:], vl))[dyn_v]
    direct = getattr(getattr(model, "module", model), "direct", False)
    r = int(rng.integers(1, pf + 1)) if pf > 0 else 0
    if r:                                                                # pushforward: the model's own rollout
        st, mv = batch["f_state"], (dyn | hinge)[:, None]
        xr, Qr, carried = pos.clone(), Q.clone(), None
        with torch.no_grad():
            for j in range(r):
                n = P + j                                                # frames 0 .. n-1 seen; frame n predicted
                xw = xr[:, :n]
                pv = torch.cat([xw[:, :1] - batch["vel"][:, :1] * DT, xw[:, :-1]], 1)
                bj = dict(batch, pos=xw, vel=(xw - pv) / DT, absent=batch["absent"][:, :n], f_ext=batch["f_ext"][:, :n],
                          f_state=st[:, :n], tau=torch.zeros(n, device=dev), frame_given=torch.ones(n, device=dev))
                fc, fe, pr = getattr(model, "module", model)(bj, lambda p_: slots(batch, p_ if carried is None else carried)[0])
                carried = pr if carried is None else carried
                m_, c_, J_ = slots(batch, carried)
                if direct:                                               # B2: vertex step, then rigid per body
                    xn = rigid_project(2 * xr[:, n - 1] - xr[:, n - 2] + fc[:, n - 1] * DT * DT, vl, body, nb, dyn_v)
                    xr[:, n] = torch.where(dyn_v[:, None], xn, xr[:, n])
                    continue
                fx = torch.where((st[:, n - 1] == F_GIVEN)[:, None], batch["f_ext"][:, n - 1], torch.zeros_like(fe[:, n - 1]))
                fx = torch.where((st[:, n - 1] == F_EMPTY)[:, None], fe[:, n - 1], fx)
                new = physics_step(batch, Qr[:, n], Qr[:, n - 1], fc[:, n - 1] + fx, xr[:, n - 1], m_, c_, J_,
                                   damping_slot(batch, carried))
                Qr[:, n + 1] = torch.where(mv, new, Qr[:, n + 1])
                xr[:, n] = torch.where(dyn_v[:, None], new[body, :3] + qrot(new[body, 3:], vl), xr[:, n])
        Qn[:, P + 1:P + 1 + r] = Qr[:, P + 1:P + 1 + r]
        pos_in[:, P:P + r] = xr[:, P:P + r]
    prev = torch.cat([pos_in[:, :1] - batch["vel"][:, :1] * DT, pos_in[:, :-1]], 1)
    b_in = dict(batch, pos=pos_in, vel=(pos_in - prev) / DT, tau=torch.cat([torch.zeros(P + r, device=dev), tau[r:]]),
                frame_given=torch.cat([torch.ones(P + r, device=dev), torch.zeros(K - r, device=dev)]))
    f_contact, f_ext_pred, pred = model(b_in, lambda pr: slots(batch, pr.detach())[0])
    m, c, J = slots(batch, pred.detach())
    state = batch["f_state"]
    f_ext = torch.where((state == F_GIVEN)[..., None], batch["f_ext"], torch.zeros_like(batch["f_ext"]))
    f_ext = torch.where((state == F_EMPTY)[..., None], f_ext_pred, f_ext)
    f_tot = f_contact + f_ext
    damping = damping_slot(batch, pred.detach())
    errs = []
    for k in range(K):                                                   # step frame P-1+k -> P+k
        if direct:                                                       # B2: x' = 2x - x_prev + a dt^2 per vertex
            x = 2 * pos_in[:, P - 1 + k] - pos_in[:, P - 2 + k] + f_contact[:, P - 1 + k] * DT * DT
        else:
            new = physics_step(batch, Qn[:, P + k], Qn[:, P + k - 1], f_tot[:, P - 1 + k], pos_in[:, P - 1 + k], m, c, J, damping)
            x = new[body, :3] + qrot(new[body, 3:], vl)
        errs.append(((x - pos[:, P + k]) ** 2).sum(-1)[dyn_v])
    err2 = torch.stack(errs, 1)                                          # squared vertex errors (m^2)
    err = (err2 + 1e-12).sqrt() * 1e-2 if l1 else err2                   # l1: distance, so that err / 1e-4 is in cm
    if w_sig > 0:
        s_k = torch.cat([torch.zeros(1, device=dev), tau[:K - 1] * sig_p * torch.arange(1, K, device=dev).sqrt()])
        s_k[:r + 1] = 0.0                                                # pushforward steps start from given frames
        w = w_sig ** 2 / (w_sig ** 2 + s_k ** 2)
        l_pos = (err.mean(0) * w).sum() / w.sum() / 1e-4
    else:
        l_pos = err.mean() / 1e-4
    if mix:
        l_pos = (err2 + 1e-12).sqrt().mean() / 1e-2 + err2.mean() / 1e-4
    hid = batch["param_has"] & ~batch["param_given"]
    cont = hid.clone(); cont[:, SOLID] = False
    l_par = ((pred - batch["params"]) ** 2)[cont].mean() if cont.any() else pred.sum() * 0
    if hid[:, SOLID].any():
        l_par = l_par + Fn.binary_cross_entropy_with_logits(pred[hid[:, SOLID], SOLID], batch["params"][hid[:, SOLID], SOLID])
    emp = (state == F_EMPTY)[:, :Fw - 1]
    l_f = (Fn.smooth_l1_loss(torch.asinh(f_ext_pred[:, :Fw - 1][emp] / FORCE_UNIT),
                             torch.asinh(batch["f_ext"][:, :Fw - 1][emp] / FORCE_UNIT)) if emp.any() else pred.sum() * 0)
    info = {"pos_rmse_mm": 1000 * err2.mean().sqrt().item(), "clean_rmse_mm": 1000 * err2[:, 0].mean().sqrt().item()}
    if r:
        info["pf_rmse_mm"] = 1000 * err2[:, 1:r + 1].mean().sqrt().item()
    return l_pos + l_par + l_f, dict(info, l_pos=l_pos.item(), l_par=l_par.item(), l_f=l_f.item())


def _load(paths, have):
    """Episodes of shards not loaded yet (a dataset still being generated grows under us)."""
    out = []
    for s in paths:
        if s not in have:
            out += [e for e in load_shard(s[:-4]) if any(b.role == "dynamic" for b in e.bodies)]
            have.add(s)
    return out


def _windows(episodes, n, P, K, rng):
    wins = []
    for _ in range(n):
        e = episodes[rng.integers(len(episodes))]
        wins.append(make_window(e, int(rng.integers(1, e.q.shape[0] - K)), P, K, rng))
    return wins


def main():
    """Single GPU, or DDP on two: torchrun --nproc_per_node 2 -m lsim.train.train ..."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/v0", help="dataset directories, comma-separated")
    ap.add_argument("--val", default="", help="held-out dataset (--split test) for rollout validation")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16, help="windows per step per GPU")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--direct", action="store_true", help="baseline B2: vertex accelerations, no physics layer")
    ap.add_argument("--arch", default="net", choices=("net", "gns"), help="gns: baseline B1 (lsim.model.gns)")
    ap.add_argument("--accum", type=int, default=1, help="batches per optimiser step (one GPU standing in for several)")
    ap.add_argument("--reload", type=int, default=1000, help="steps between looking for new shards")
    ap.add_argument("--out", default="outputs/lsim/train_v0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--pair_norm", action="store_true", help="contact forces per body pair (Simulator pair_norm)")
    ap.add_argument("--support", action="store_true", help="default support on static contacts (Simulator support)")
    ap.add_argument("--coulomb", action="store_true", help="default Coulomb contact on static surfaces (support='coulomb')")
    ap.add_argument("--pf", type=int, default=0, help="pushforward: up to this many own-rollout frames (step_loss pf)")
    ap.add_argument("--l1", action="store_true", help="position loss on distances, not squared (step_loss l1)")
    ap.add_argument("--mix", action="store_true", help="position loss: distance + squared distance (step_loss mix)")
    ap.add_argument("--noise", default="0.01,0.05", help="chunk noise per frame: m, rad (step_loss sig_p, sig_r)")
    ap.add_argument("--hist_noise", default="0,0", help="lo,hi m/frame: unflagged history noise (step_loss sig_h)")
    ap.add_argument("--step_weight", type=float, default=0.0, help="m: noise-normalised step weights (step_loss w_sig)")
    a = ap.parse_args()
    noise = dict(zip(("sig_p", "sig_r"), (float(x) for x in a.noise.split(","))),
                 sig_h=tuple(float(x) for x in a.hist_noise.split(",")), w_sig=a.step_weight, pf=a.pf, l1=a.l1, mix=a.mix)
    import os
    from concurrent.futures import ThreadPoolExecutor
    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda", torch.cuda.current_device())
    log = print if rank == 0 else (lambda *x, **k: None)
    have = set()
    def shard_paths():
        return sorted(p for d in a.data.split(",") for p in glob.glob(f"{d}/shard_*.npz"))
    episodes = _load(shard_paths(), have)
    log(f"{len(episodes)} episodes from {len(have)} shards", flush=True)
    val = []
    if a.val and rank == 0:
        vrng = np.random.default_rng(7)
        vep = _load(sorted(glob.glob(f"{a.val}/shard_*.npz")), set())
        val = [collate([w], dev, np.random.default_rng(100 + i), p_hist_noise=0.0, augment=False)
               for i, w in enumerate(_windows(vep, 64, 8, 8, vrng))]
    core = build_model(vars(a), len(PARAMS)).to(dev)
    model = torch.nn.parallel.DistributedDataParallel(core, find_unused_parameters=True) if ddp else core
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    start = 0
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"] + 1
    rng = np.random.default_rng(start * 7919 + rank)
    P, K = 8, 8
    pool = ThreadPoolExecutor(1)                                     # the next batch is built while this one trains
    nxt = pool.submit(_windows, episodes, a.batch, P, K, np.random.default_rng(rng.integers(1 << 31)))
    t0 = time.time()
    for step in range(start, a.steps):
        wins = nxt.result()
        if a.reload and step % a.reload == 0 and step > start:
            new = _load(shard_paths(), have)
            if new:
                episodes = episodes + new
                log(f"[data] {len(episodes)} episodes from {len(have)} shards", flush=True)
        opt.zero_grad(set_to_none=True)
        for micro in range(a.accum):
            if micro:
                wins = nxt.result()
            nxt = pool.submit(_windows, episodes, a.batch, P, K, np.random.default_rng(rng.integers(1 << 31)))
            batch = collate(wins, dev, rng)
            loss, info = step_loss(model, batch, rng, K, dev, **noise)
            (loss / a.accum).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            log(f"step {step:6d}  loss {loss.item():.3e}  " + "  ".join(f"{k} {v:.3g}" for k, v in info.items()) +
                f"  grad {gn.item():.2g}  lr {sched.get_last_lr()[0]:.2g}  {time.time()-t0:.0f} s", flush=True)
        if val and (step % 1000 == 0 or step == a.steps - 1):
            core.eval()
            with torch.no_grad():
                r = [step_loss_ar(core, b, rng, K, dev)[1] for b in val]
            core.train()
            log(f"[val] step {step}  8-frame rollout on held-out objects: first {np.median([x['rmse_first_mm'] for x in r]):.2f} mm, "
                f"last {np.median([x['rmse_last_mm'] for x in r]):.2f} mm (median of 64 windows), "
                f"p90 last {np.percentile([x['rmse_last_mm'] for x in r], 90):.1f} mm; resting bodies last "
                f"{np.nanmedian([x['rest_last_mm'] for x in r]):.2f} mm", flush=True)
        if rank == 0 and (step % 2000 == 0 or step == a.steps - 1):
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "args": vars(a), "step": step}, out / "last.pt")
    if ddp:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
