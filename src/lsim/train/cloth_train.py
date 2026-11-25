"""Training the cloth path (phase 3): lsim.model.cloth_net on cloth windows (lsim.train.cloth_windows).

As the rigid trainer (lsim.train.train.step_loss): every chunk step is ONE physics step (cloth_step) from the
window's own state at the frame before it, scored against the true next frame. The cloth's free vertices get a
random-walk noise over the chunk frames (sig m per frame, scaled by each frame's diffusion-forcing level tau), so
the model learns to pull a perturbed cloth back; held vertices, the gripper and static points stay true (given).
Loss: distance + squared distance of the free cloth vertices (cm, cm^2: linear below ~5 mm), plus the hidden slots;
with edge_w > 0 also the error of each mesh edge's length against the teacher's next frame, over its rest length
(the teacher's towels strain 1e-4, the model's stretched 1% in one step: neighbours' relative error is a small part of
the position loss, and it compounds). Pushforward (pf > 0): the first r chunk frames are the model's own rollout,
flagged given (Brandstetter et al.).

--gen energy (2026-11-02, with --noise_dim > 0: a generative network, lsim.model.cloth_qwen): the position losses become
the energy score of two samples drawn from the same input, per free vertex (|X1 - y| + |X2 - y| - |X1 - X2|) / 2 in
cm -- a proper scoring rule, minimised by the distribution of futures rather than their mean, the fair CRPS of
CRPS-trained weather models; on the long rollouts two independent rollouts from the same start, scored each frame.

    $L -m lsim.train.cloth_train --data data/datasets/c1 --steps 20000 --out outputs/lsim/cloth_v0
"""
import os
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from lsim.model.cloth_net import G, PUSH_OUT, build_cloth_model, cloth_step, decode, direct_step, kin_next, step_frames
from lsim.train.cloth_windows import (F_EMPTY, F_GIVEN, MATERIAL, collate_cloth, load_cloth_shard,  # noqa: F401
                                     load_shard, make_cloth_window)

DT = 1.0 / float(os.environ.get("LSIM_FPS", 30.0))   # one frame (LSIM_FPS, as lsim.model.cloth_net)
RATE = (30.0 * DT) ** 2 if os.environ.get("LSIM_FPS") else 1.0   # a step's displacement per unit force against 30 fps'
LOG_CHI3 = 0.5 * (0.03649 + float(np.log(2.0)))   # E[log |e|] - log sigma for an isotropic 3D Gaussian: (psi(3/2) + ln 2) / 2


def forces(batch, f_int, f_con, w):
    """Network forces in newtons: (internal + contact) x the vertex weight."""
    m = w[batch["body_of"]] * batch["area"]
    return (f_int + f_con) * (m * G)[:, None, None]


def contact_kw(core, batch, f_con, w):
    """What the exact layer takes besides the total force, by the layer the model trains with (ClothSimulator.layer):
    layer 2 counts the network's contact force toward a surface's normal impulse and pushes vertices out of planes."""
    if getattr(core, "layer", 1) >= 2:
        kw = {"f_contact": forces(batch, 0.0, f_con, w), "push_out": PUSH_OUT}
    else:
        kw = {"f_contact": None, "push_out": 0.0}
    if getattr(core, "restore", None) is not None:                     # shape restoration (cloth_net.restore_shape)
        kw["restore"] = core.restore
    return kw


CLOTH_EDGE = 0.01                                  # m: the cloth's median rest edge (c1: every cloth h = 1 cm)


def noise_scale(batch):
    """Per vertex (N,): the chunk noise's scale for its body under --noise_rel -- the body's median rest edge over
    the cloth's (CLOTH_EDGE), so every material's inputs are strained alike by the noise (a soft body's 3-5 mm tet
    edges took 1 mm per vertex as 20-30% strain); exactly 1 for a cloth, so cloth is unchanged."""
    E = batch["edges"]
    L0 = (batch["rest"][E[:, 0]] - batch["rest"][E[:, 1]]).norm(dim=-1) * batch["size"][batch["sample_of"][E[:, 0]]]
    eb = batch["body_of"][E[:, 0]]
    nb = int(batch["n_bodies"])
    f = torch.ones(nb, device=L0.device)
    mat = batch.get("material")
    for k in range(nb):
        if mat is not None and int(mat[k]) != 0 and bool((eb == k).any()):
            f[k] = L0[eb == k].median() / CLOTH_EDGE
    return f[batch["body_of"]]


def chunk_inputs(batch, pos, free, kn, P, K, dev):
    """--chunk: the network's inputs when it predicts all K frames from one call -- the P true past frames, and for the
    K coming frames everything given (gripper path, held vertices, static points) but no cloth positions: each free
    vertex continues at its last velocity (a placeholder carrying nothing the past does not), every coming frame
    marked not given and fully noised (tau 1)."""
    assert pos.shape[1] == P + K, (pos.shape, P, K)
    steps = torch.arange(1, K + 1, device=dev, dtype=pos.dtype)[None, :, None]
    cv = pos[:, P - 1:P] + (pos[:, P - 1:P] - pos[:, P - 2:P - 1]) * steps
    pos_in = torch.cat([pos[:, :P], torch.where(free[:, P - 1:P - 1 + K, None], cv, pos[:, P:P + K])], 1)
    prev = torch.cat([pos_in[:, :1] - batch["vel"][:, :1] * DT, pos_in[:, :-1]], 1)
    return dict(batch, pos=pos_in, vel=(pos_in - prev) / DT, tau=torch.cat([torch.zeros(P, device=dev), torch.ones(K, device=dev)]),
                frame_given=torch.cat([torch.ones(P, device=dev), torch.zeros(K, device=dev)]), kin_next=kn)


def chunk_steps(batch, pos, F, kw, w, free, mu, P, K):
    """--chunk: the K steps integrated in order from the true frames P - 2, P - 1, each from the previously integrated
    positions (open loop inside the chunk; the network saw none of them) -> (N, K, 3)."""
    fc = kw.pop("f_contact")
    xs = [pos[:, P - 2], pos[:, P - 1]]
    for k in range(K):
        jj = P - 1 + k
        xs.append(cloth_step(batch, xs[-2], xs[-1], F[:, jj], w, ~free[:, jj], pos[:, jj + 1], mu,
                             f_contact=None if fc is None else fc[:, jj], **kw))
    return torch.stack(xs[2:], 1)


def jit_view(b, s):
    """The network's view of a batch with every cloth vertex of every frame moved by iid Gaussian s m per axis, the
    velocities re-differenced from the moved positions (--in_jit; the exact layer still integrates the unmoved state).
    2026-11-21: the trained network reads the clean frames' 1e-4 edge strains (micrometres on a centimetre edge) through
    asinh(1e5 strain) -- worth ~0.04 of its one-frame accuracy, absent from real data and from its own rollouts
    (scripts/l2_input_jitter.py); the teacher from 3 frames jittered 10-30 um loses nothing (0.25 -> 0.26)."""
    if s <= 0:
        return b
    cl = b["is_cloth"]
    p = b["pos"].clone()
    p[cl] = p[cl] + s * torch.randn_like(p[cl])
    prev = torch.cat([b["pos"][:, :1] - b["vel"][:, :1] * DT, p[:, :-1]], 1)
    return dict(b, pos=p, vel=(p - prev) / DT)


def step_loss(model, batch, rng, K, dev, sig=1e-3, pf=0, mix=True, edge_w=0.0, noise="vertex", sig_rot=3e-3,
              sig_white=2e-5, noise_rel=False, gen="", chunk=None, parts=None, in_jit=0.0):
    """parts: a dict to receive the loss terms as tensors (l_pos, l_edge, l_par, l_unc). chunk: K frames from one call (chunk_inputs / chunk_steps); None takes the model's own (build_cloth_model's
    m.chunk, from --chunk), 0 the one-step-per-frame training."""
    P = batch["n_past"]
    pos = batch["pos"]
    N, Fw, _ = pos.shape
    cl = batch["is_cloth"]
    free = cl[:, None] & ~batch["held"]                                 # (N, Fw): moved by the physics layer j -> j+1
    kn = kin_next(batch, torch.cat([pos, pos[:, -1:]], 1))             # the action over each step (given)
    tau = torch.sigmoid(torch.randn(K, device=dev))
    if noise == "rigid":                                                # each cloth moved as a body (random walks of a
        so = batch["sample_of"]                                         # translation, sig m, and a rotation about its
        S = int(so.max()) + 1                                           # centroid, sig_rot rad, per frame) plus white
        cnt = torch.zeros(S, device=dev).index_add_(0, so[cl], torch.ones(int(cl.sum()), device=dev)).clamp_min(1)
        c = torch.zeros(S, 3, device=dev).index_add_(0, so[cl], pos[cl, P - 1]) / cnt[:, None]   # noise, sig_white:
        tw = torch.cumsum(torch.randn(S, K, 3, device=dev) * sig, 1)   # the rigid recipe's noise in kind. Per-vertex
        rw = torch.cumsum(torch.randn(S, K, 3, device=dev) * sig_rot, 1)   # noise at 1 mm strains 1 cm edges by
        walk = tw[so] + torch.cross(rw[so], pos[:, P:P + K] - c[so][:, None], dim=-1)   # 10-20%, 1,000x the teacher's
        walk = (walk + torch.cumsum(torch.randn(N, K, 3, device=dev) * sig_white, 1)) * tau[None, :, None]
    elif noise_rel:                                                     # per material, by its edge length
        sig_v = sig * noise_scale(batch)[:, None, None]
        walk = torch.cumsum(torch.randn(N, K, 3, device=dev) * sig_v, 1) * tau[None, :, None]
    else:
        walk = torch.cumsum(torch.randn(N, K, 3, device=dev) * sig, 1) * tau[None, :, None]
    noised = free[:, P - 1:P - 1 + K]                                   # a vertex free in the step INTO the frame
    pos_in = pos.clone()
    pos_in[:, P:P + K] = pos[:, P:P + K] + walk * noised[..., None]
    r = int(rng.integers(1, pf + 1)) if pf > 0 else 0
    if r:                                                                # pushforward: the model's own rollout
        with torch.no_grad():
            xr = pos.clone()
            for j in range(r):
                n = P + j
                xw = xr[:, :n]
                pv = torch.cat([xw[:, :1] - batch["vel"][:, :1] * DT, xw[:, :-1]], 1)
                bj = dict(batch, pos=xw, vel=(xw - pv) / DT, absent=batch["absent"][:, :n], f_ext=batch["f_ext"][:, :n],
                          f_state=batch["f_state"][:, :n], held=batch["held"][:, :n], tau=torch.zeros(n, device=dev),
                          frame_given=torch.ones(n, device=dev), kin_next=kn[:, :n])
                fi, fc, pr, _ = getattr(model, "module", model)(jit_view(bj, in_jit))   # unwrapped: DDP wants one forward
                if getattr(getattr(model, "module", model), "direct", False):  # per backward
                    xr[:, n] = direct_step(xr[:, n - 2], xr[:, n - 1], fi[:, n - 1], ~free[:, n - 1], pos[:, n])
                    continue
                w, mu = decode(pr, batch)
                F = forces(batch, fi, fc, w)
                kw = contact_kw(getattr(model, "module", model), batch, fc, w)
                kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, n - 1]
                xr[:, n] = cloth_step(batch, xr[:, n - 2], xr[:, n - 1], F[:, n - 1], w, ~free[:, n - 1], pos[:, n], mu, **kw)
        pos_in[:, P:P + r] = xr[:, P:P + r]
    prev = torch.cat([pos_in[:, :1] - batch["vel"][:, :1] * DT, pos_in[:, :-1]], 1)
    b_in = dict(batch, pos=pos_in, vel=(pos_in - prev) / DT, tau=torch.cat([torch.zeros(P + r, device=dev), tau[r:]]),
                frame_given=torch.cat([torch.ones(P + r, device=dev), torch.zeros(K - r, device=dev)]), kin_next=kn)
    j = torch.arange(P - 1, P - 1 + K, device=dev)                      # all K steps at once: frame j -> j + 1
    core = getattr(model, "module", model)
    if chunk is None:
        chunk = getattr(core, "chunk", 0)
    if chunk:                                                           # one call for the K frames (the noise and
        b_in = chunk_inputs(batch, pos, free, kn, P, K, dev)            # pushforward above go unused)

    def predict():
        f_int, f_con, pred, _ = model(jit_view(b_in, in_jit))
        w, mu = decode(pred.detach(), batch)                            # predicted slots reach physics detached
        F = forces(batch, f_int, f_con, w)
        if getattr(core, "direct", False):
            return direct_step(pos_in[:, j - 1], pos_in[:, j], f_int[:, j], ~free[:, j], pos[:, j + 1]), pred
        kw = contact_kw(core, batch, f_con, w)
        if chunk:
            return chunk_steps(batch, pos, F, kw, w, free, mu, P, K), pred
        kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, j]
        return step_frames(batch, pos_in[:, j - 1], pos_in[:, j], F[:, j], w, ~free[:, j], pos[:, j + 1], mu, **kw), pred
    x, pred = predict()
    ls_all = core.log_sigma if getattr(core, "unc", None) is not None else None
    lv_all = core.slot_logvar if getattr(core, "prob_slots", False) else None
    fr = free[:, j]
    err2 = ((x - pos[:, j + 1]) ** 2).sum(-1)[fr]
    l_unc = torch.zeros((), device=dev)
    if ls_all is not None:                                              # the head reads E[log |error|]: regression in
        ls = ls_all[:, j][fr]                                            # log space, bounded (a Gaussian likelihood
        l_unc = ((ls - 0.5 * torch.log(err2.detach() + 1e-14)) ** 2).mean()   # gave gradients of 1e6 at the start,
                                                                         # which global clipping took from the forces)
    k_of = torch.arange(K, device=dev)[None].expand(N, -1)[fr]
    x2 = None
    if gen == "energy":                                                 # a second sample from the same input: the
        x2, _ = predict()                                               # energy score, per free vertex (cm)
        d1 = (err2 + 1e-12).sqrt()
        d2 = (((x2 - pos[:, j + 1]) ** 2).sum(-1)[fr] + 1e-12).sqrt()
        d12 = (((x - x2) ** 2).sum(-1)[fr] + 1e-12).sqrt()
        l_pos = (0.5 * (d1 + d2).mean() - 0.5 * d12.mean()) / 1e-2
    elif gen:
        raise ValueError(f"--gen {gen}: only 'energy'")
    else:
        # RATE (1 at 30 fps): at LSIM_FPS a force error moves a vertex RATE times as far in a step, so the position
        # and edge terms are taken per that displacement and weigh against the side terms as at 30 fps
        l_pos = (((err2 + 1e-12).sqrt().mean() / (1e-2 * RATE) + err2.mean() / (1e-4 * RATE * RATE)) if mix
                 else err2.mean() / (1e-4 * RATE * RATE))
    E = batch["edges"]                                                  # the mesh edges' lengths against the teacher's
    ei, ej = E[:, 0], E[:, 1]
    L0 = (batch["rest"][ei] - batch["rest"][ej]).norm(dim=-1) * batch["size"][batch["sample_of"][ei]]
    tgt = pos[:, P:P + K]
    d_len = ((x[ei] - x[ej]).norm(dim=-1) - (tgt[ei] - tgt[ej]).norm(dim=-1)).abs() / L0[:, None]   # (edges, K)
    if x2 is not None:                                                  # both samples' edges
        d_len = 0.5 * (d_len + ((x2[ei] - x2[ej]).norm(dim=-1) - (tgt[ei] - tgt[ej]).norm(dim=-1)).abs() / L0[:, None])
    d_len = d_len[fr[ei] | fr[ej]]
    l_edge = edge_w * d_len.mean() / RATE if edge_w > 0 else torch.zeros((), device=dev)   # weight 1: its gradient on a
                                                                         # vertex ~ the position loss's at 1 mm
    hid = ~batch["param_given"]
    if getattr(core, "prob_slots", False):                              # Gaussian negative log likelihood
        lv = lv_all
        nll = 0.5 * (lv + (pred - batch["params"]) ** 2 * torch.exp(-lv))
        l_par = nll[hid].mean() if hid.any() else pred.sum() * 0
    else:
        l_par = ((pred - batch["params"]) ** 2)[hid].mean() if hid.any() else pred.sum() * 0
    first = err2[k_of == 0]
    info = {"pos_rmse_mm": 1000 * err2.mean().sqrt().item(), "clean_rmse_mm": 1000 * first.mean().sqrt().item() if len(first) else float("nan"),
            "clean_mean_mm": 1000 * first.sqrt().mean().item() if len(first) else float("nan"),
            "l_pos": l_pos.item(), "l_par": l_par.item(), "strain_err_pct": 100 * d_len.mean().item()}
    if edge_w > 0:
        info["l_edge"] = l_edge.item()
    if x2 is not None:                                                  # calibration: E|X1 - X2| = E|X - y| if the
        info["es_err_mm"] = 1e3 * d1.mean().item()                      # samples and the truth come from one law
        info["es_spread_mm"] = 1e3 * d12.mean().item()
    if hid.any():
        info["slot_err"] = ((pred - batch["params"]).abs()[hid]).mean().item()
    if hasattr(core, "material") and "material" in batch:            # ClothQwen3D: each material's errors apart
        mat = batch["material"][batch["body_of"]][:, None].expand(-1, K)[fr]
        for name, m in MATERIAL.items():
            sel = mat == m
            if sel.any():
                info[f"{name}_rmse_mm"] = 1000 * err2[sel].mean().sqrt().item()
                f1 = sel & (k_of == 0)
                if f1.any():
                    info[f"{name}_clean_mm"] = 1000 * err2[f1].sqrt().mean().item()
                info[f"{name}_l_pos"] = (((err2[sel] + 1e-12).sqrt().mean() / 1e-2 + err2[sel].mean() / 1e-4) if mix
                                         else err2[sel].mean() / 1e-4).item()
    if getattr(core, "unc", None) is not None:
        info["l_unc"] = l_unc.item()
        info["sigma_mm"] = 1e3 * torch.exp(ls_all[:, j][fr] - LOG_CHI3).median().item()
    if parts is not None:
        parts.update(l_pos=l_pos, l_edge=l_edge, l_par=l_par, l_unc=l_unc, x=x.detach(), free=fr, err2=err2)   # x: predicted (N, K, 3)
    return l_pos + l_edge + l_par + l_unc, info


@torch.no_grad()
def rollout(model, ep, t0, n_frames, P=8, dev="cuda", hidden=(), rng=None, sample=False, chaos=0.0, seed=0):
    """Roll the model out frame by frame for n_frames from frame t0 of a cloth episode, from its true history; the
    gripper, held vertices and static points follow the recording. sample: draw hidden slots from the slot posterior
    (prob_slots models) instead of taking its mean; chaos (m): Gaussian noise added to the free vertices every
    frame, the teacher's own float-order noise, so that an ensemble spreads as the teacher does. Returns (pred, true)
    cloth positions (n_frames, Vc, 3) and the slots used."""
    rng = np.random.default_rng(0) if rng is None else rng
    win = make_cloth_window(ep, t0 + n_frames - 1, P + n_frames - 1, 1, np.random.default_rng(0))
    # the window's frames: t0 - P + 1 .. t0 + n_frames (P + n_frames frames); build once, overwrite as we go
    batch = collate_cloth([win], dev, rng, p_hidden=0.0, augment=False, all_past=True, hidden=hidden)
    Vc = win["n_cloth"]
    pos = batch["pos"].clone()
    true = pos.clone()
    free = batch["is_cloth"][:, None] & ~batch["held"]
    carried = None
    for k in range(n_frames):
        n = P + k                                                        # frames 0 .. n-1 known, n predicted
        lo = n - P                                                       # the model sees the last P frames
        xw = pos[:, lo:n]
        pv = pos[:, lo - 1:n - 1] if lo > 0 else torch.cat([xw[:, :1] - batch["vel"][:, :1] * DT, xw[:, :-1]], 1)
        bj = dict(batch, pos=xw, vel=(xw - pv) / DT, absent=batch["absent"][:, lo:n], f_ext=batch["f_ext"][:, lo:n],
                  f_state=batch["f_state"][:, lo:n], held=batch["held"][:, lo:n], tau=torch.zeros(P, device=dev),
                  frame_given=torch.ones(P, device=dev), n_past=P)
        bj["kin_next"] = kin_next(bj, true[:, lo:n + 1])
        if "vtrue" in batch:                                             # --oracle_v: only the recorded frames' own
            bj["vtrue"] = batch["vtrue"][:, lo:n]
            bj["vtrue_on"] = (torch.arange(lo, n, device=dev) < P)[None].expand(pos.shape[0], -1)
        fi, fc, pr, _ = model(bj)
        if carried is None:                                              # slots from the first window, kept
            carried = pr
            if sample and getattr(model, "prob_slots", False):
                g = torch.Generator(device=pr.device).manual_seed(seed)
                carried = pr + torch.exp(0.5 * model.slot_logvar) * torch.randn(pr.shape, generator=g, device=pr.device)
        w, mu = decode(carried, batch)
        if getattr(model, "direct", False):
            pos[:, n] = direct_step(pos[:, n - 2], pos[:, n - 1], fi[:, -1], ~free[:, n - 1], true[:, n])
        else:
            F = forces(bj, fi, fc, w)
            kw = contact_kw(model, bj, fc, w)
            kw["f_contact"] = None if kw["f_contact"] is None else kw["f_contact"][:, -1]
            pos[:, n] = cloth_step(batch, pos[:, n - 2], pos[:, n - 1], F[:, -1], w, ~free[:, n - 1], true[:, n], mu, **kw)
        if chaos != 0:                                                   # a fixed level, or the learned one ("unc")
            g = torch.Generator(device=pos.device).manual_seed(seed * 1000 + k)
            sig = torch.exp(model.log_sigma[:, -1] - LOG_CHI3)[:, None] if chaos == "unc" else chaos
            pos[:, n] += torch.randn(pos[:, n].shape, generator=g, device=pos.device) * sig * free[:, n - 1, None]
    return pos[:Vc, P:P + n_frames].transpose(0, 1).cpu().numpy(), true[:Vc, P:P + n_frames].transpose(0, 1).cpu().numpy(), carried


@torch.no_grad()
def validate(model, val, dev, K=8):
    """Held-out windows (fixed): the clean first step's mean free-vertex error, and 10-frame rollouts from a few of
    them (mm, median over windows), with the network's outputs as they are."""
    core = getattr(model, "module", model)
    was = core.training
    core.eval()
    one, roll = [], []
    for i, b in enumerate(val["batches"]):
        _, info = step_loss(core, b, np.random.default_rng(i), K, dev, sig=0.0)
        one.append(info["clean_mean_mm"])
    for e, t0 in val["starts"]:
        pred, true, _ = rollout(core, e, t0, 10, dev=dev)
        free = ~e.held[t0:t0 + 10]
        d = np.linalg.norm(pred - true, axis=-1)
        roll.append(1e3 * float(np.mean([d[j][free[j]].mean() for j in range(10) if free[j].any()])))
    core.train(was)
    return float(np.median(one)), float(np.median(roll))


class ZeroForces(torch.nn.Module):
    """The exact layer alone: no network forces, the given slots (a reference for the rollouts)."""
    layer, prob_slots, direct = 2, False, False

    def forward(self, batch):
        z = torch.zeros_like(batch["pos"])
        return z, z, batch["params"], None


def resting_starts(episodes, n=8, span=(8, 10), still=5e-4, seed=0):
    """(episode, t0) where nothing is held and no vertex moves more than `still` m over the span frames before and
    after t0 (a body at rest on the table, nothing touching it), up to n of them."""
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(episodes)):
        e = episodes[i]
        x = e.x(np.arange(e.n_frames))
        held = e.held.any(1)
        for t0 in range(span[0], e.n_frames - span[1] - 1, 6):
            sl = slice(t0 - span[0], t0 + span[1] + 1)
            if not held[sl].any() and np.linalg.norm(x[sl] - x[t0], axis=-1).max() < still:
                out.append((e, t0))
                break
        if len(out) >= n:
            break
    return out


def resting_drift(model, starts, dev, n=10):
    """How far a resting body's vertices move in n frames under model (mm): the median over starts of the mean and of
    the largest vertex displacement from the start."""
    mean, mx = [], []
    for e, t0 in starts:
        pred, _, _ = rollout(model, e, t0, n, dev=dev)
        d = np.linalg.norm(pred[-1] - e.x([t0])[0], axis=-1)
        mean.append(1e3 * d.mean()); mx.append(1e3 * d.max())
    return float(np.median(mean)), float(np.median(mx))


def val_choice(spec):
    """--val as entries (comma-separated), each a glob or alternatives 'A|B' (the first that matches shards: e.g.
    s1_test once it exists, else the soft pilot): the chosen globs."""
    out = []
    for ent in spec.split(","):
        alts = ent.split("|")
        out.append(next((g for g in alts if glob.glob(g)), alts[-1]))
    return tuple(out)


def make_val_kinds(choice, dev, log=print):
    """Held-out windows and rollout starts per material (one --val entry each), as the single-glob validation builds
    them; for soft bodies also resting starts and their drift with the network's outputs at zero (the exact layer
    alone: a soft body's inside is held only by learned forces)."""
    out = {"choice": choice, "kinds": []}
    for gi, g in enumerate(choice):
        paths = sorted(glob.glob(g))[:4]
        if not paths:
            log(f"[val] {g}: no shards yet", flush=True)
            continue
        vrng = np.random.default_rng(12345)
        veps = [e for p in paths for e in load_shard(p[:-4])]
        kind = getattr(veps[0], "material", "cloth")
        vw = _windows(veps, 64, 8, 8, vrng)
        v = {"kind": kind, "batches": [collate_cloth(vw[i:i + 4], dev, np.random.default_rng(i), p_hidden=0.0,
                                                     augment=False, all_past=True) for i in range(0, 64, 4)],
             "starts": [(veps[k], int(vrng.integers(8, veps[k].n_frames - 11)))
                        for k in vrng.choice(len(veps), min(16, len(veps)), replace=False)]}
        if kind == "soft":
            v["rest"] = resting_starts(veps)
            if v["rest"]:
                v["rest_zero"] = resting_drift(ZeroForces(), v["rest"], dev)
        log(f"[val] {kind}: {len(veps)} held-out episodes from {g}: 64 windows, {len(v['starts'])} rollout starts"
            + (f", {len(v['rest'])} resting starts (zero output: drift {v['rest_zero'][0]:.2f} mm mean, "
               f"{v['rest_zero'][1]:.2f} mm max in 10 frames)" if v.get("rest") else ""), flush=True)
        out["kinds"].append(v)
    return out


def validate_kinds(model, val, dev, step, log=print):
    """validate per material, plus the soft bodies' resting drift under the model; one log line per material."""
    core = getattr(model, "module", model)
    for v in val["kinds"]:
        one, roll = validate(model, v, dev)
        msg = f"[val] step {step}  {v['kind']}: held-out clean step {one:.3f} mm  10-frame rollout {roll:.2f} mm (median)"
        if v.get("rest"):
            was = core.training
            core.eval()
            with torch.no_grad():
                dm, dx = resting_drift(core, v["rest"], dev)
            core.train(was)
            msg += f"  resting drift 10 frames {dm:.2f} mm mean / {dx:.2f} max (zero output {v['rest_zero'][0]:.2f})"
        log(msg, flush=True)


def _load(paths, have, origin=None):
    """Episodes of the shards in paths not yet in have (any material: cloth_windows.load_shard). origin: a list to
    append each new episode's source shard path to."""
    out = []
    for s in paths:
        if s not in have:
            new = load_shard(s[:-4])
            out += new
            if origin is not None:
                origin += [s] * len(new)
            have.add(s)
    return out


def _windows(episodes, n, P, K, rng, groups=None, weights=None, p_hide_grasp=None):
    """n training windows from random episodes: uniform over episodes, or (groups: episode indices per dataset,
    weights: their probabilities) a dataset by its weight, then an episode of it uniformly."""
    wins = []
    for _ in range(n):
        if groups is None:
            e = episodes[rng.integers(len(episodes))]
        else:
            g = groups[int(rng.choice(len(groups), p=weights))]
            e = episodes[g[rng.integers(len(g))]]
        wins.append(make_cloth_window(e, int(rng.integers(1, e.n_frames - K)), P, K, rng,
                                      **({} if p_hide_grasp is None else {"p_hide_grasp": p_hide_grasp})))
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets/c1", help="dataset directories, comma-separated (any of cloth, "
                    "rope, soft shards: cloth_windows.load_shard)")
    ap.add_argument("--data_w", default="", help="sampling weights of the --data datasets, comma-separated (e.g. "
                    "0.5,0.25,0.25): a window's dataset by its weight, then an episode of it uniformly; empty: uniform "
                    "over all episodes")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--arch", default="graph", choices=("graph", "qwen", "qwen3d"),
                    help="graph: ClothSimulator (message passing + attention); qwen: lsim.model.cloth_qwen.ClothQwen; "
                         "qwen3d: ClothQwen3D (cloth, rope and soft bodies: 3D rest positions, a material embedding)")
    ap.add_argument("--heads", type=int, default=4, help="qwen: attention heads")
    ap.add_argument("--kv_heads", type=int, default=2, help="qwen: key / value heads (grouped-query)")
    ap.add_argument("--mlp", type=int, default=0, help="qwen: SwiGLU hidden width (0: 3 d)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sig", type=float, default=1e-3, help="m per frame: chunk noise on free cloth vertices")
    ap.add_argument("--pf", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=0, help="8: the network predicts the window's 8 coming frames from ONE "
                    "call (the past 8 frames true, the coming ones' cloth positions unseen), the exact layer integrates "
                    "them in order (open loop); 0: one step per frame from the (noised) history (2026-11-06)")
    ap.add_argument("--shards", type=int, default=0, help="use only the first n shards (overfit tests)")
    ap.add_argument("--episodes", type=int, default=0, help="use only the first n episodes (overfit tests)")
    ap.add_argument("--own", action="store_true", help="qwen, cloth: every vector a cloth token reads and gives in the "
                    "vertex's own frame (cloth_qwen._forward_own): a rotated scene is the same input")
    ap.add_argument("--own_up", action="store_true", help="with --own: also the world-z part of each vector a cloth token "
                    "reads, and a force along world z from its own head (held-out cloth: --own alone did not learn to "
                    "cancel gravity)")
    ap.add_argument("--no_aug", action="store_true", help="no random yaw and shift per window (the small-fit test, "
                    "2026-11-22: 8 windows of one episode fit to 0.06 of constant velocity as they are, 0.71 re-yawed "
                    "every step)")
    ap.add_argument("--all_past", action="store_true", help="every window with its 8 given frames (no random shorter past)")
    ap.add_argument("--given", action="store_true", help="every input given (the user, 2026-11-24: start with all inputs given): "
                    "no parameter slot hidden, the grasp force never hidden, all 8 past frames")
    ap.add_argument("--random_past", action="store_true", help="with --given: keep the random shorter past (slots and "
                    "grasp force still given) -- splits --given's rollout cost (2026-11-24)")
    ap.add_argument("--subset", default="", help="a JSON file {\"episodes\": [[shard stem, world], ...]}: train on these "
                    "episodes only (the coverage study, scripts/l1_coverage_subsets.py); no shard reloading")
    ap.add_argument("--out", default="outputs/lsim/cloth_v0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--side_apart", action="store_true", help="qwen: the side heads cannot touch the force path -- the "
                    "slot head reads the trunk detached (the uncertainty head already does) and both are clipped apart "
                    "from the rest (their gradients otherwise share the one norm clip_grad_norm_ scales to 1)")
    ap.add_argument("--in_jit", type=float, default=0.0, help="m: the network sees every cloth vertex of every frame "
                    "moved by iid Gaussian noise of this size per axis, training only (jit_view; 1e-5: no information "
                    "below ~10 um, where real data and the model's own rollouts have none)")
    ap.add_argument("--drop_fine", action="store_true", help="qwen: no asinh(1e5 strain) neighbour feature (zeroed; "
                    "the micrometre stretch reading, 2026-11-21)")
    ap.add_argument("--fps", type=float, default=30.0, help="the data's frame rate; must equal the environment's LSIM_FPS "
                    "(read at import by the exact layer and the windows; frame-rate study), kept in the checkpoint")
    ap.add_argument("--oracle_v", action="store_true", help="qwen, diagnostic: the teacher's instantaneous velocities at the "
                    "clean past frames as extra inputs (needs record_v data, l1_generate_cloth --record_v)")
    ap.add_argument("--patch", type=int, default=0, help="qwen, cloth: each vertex's (2R+1)^2 grid stencil in its own frame "
                    "(deviation from the flat grid, relative velocity) as inputs, and a force head in that frame (2026-11-19: "
                    "a linear map of the 9x9 stencil predicts the next frame better than the 64M network)")
    ap.add_argument("--fp32", action="store_true", help="the network in fp32 throughout, no bf16 autocast (precision test)")
    ap.add_argument("--no_ckpt", action="store_true", help="keep activations (faster backward, more memory)")
    ap.add_argument("--no_global", action="store_true", help="no per-frame attention across the cloth (ablation)")
    ap.add_argument("--prob_slots", action="store_true", help="Gaussian slot posterior (ClothSimulator prob_slots)")
    ap.add_argument("--direct", action="store_true", help="baseline B2: whole accelerations, no exact layer")
    ap.add_argument("--unc", action="store_true", help="one-step uncertainty head (ClothSimulator unc)")
    ap.add_argument("--accel", action="store_true", help="accelerations as inputs (ClothSimulator accel)")
    ap.add_argument("--layer", type=int, default=2, help="the exact layer's contact default (ClothSimulator layer)")
    ap.add_argument("--edge", type=float, default=0.0, help="weight of the mesh-edge length loss (step_loss edge_w)")
    ap.add_argument("--restore", default="", help="the exact layer's edge-length / tet-volume restoration "
                    "(cloth_net.restore_shape; 2026-11-03): '' off, '1' the teachers' bands (cloth_net.RESTORE_BANDS), "
                    "or a cloth_net.restore_config spec ('cloth=0.01:0.01,iters=10,...')")
    ap.add_argument("--noise", default="vertex", choices=("vertex", "rigid"),
                    help="chunk noise: per-vertex random walk (sig), or each cloth moved as a body (sig, --sig_rot) "
                         "plus per-vertex --sig_white")
    ap.add_argument("--sig_rot", type=float, default=3e-3, help="rad per frame (--noise rigid)")
    ap.add_argument("--sig_white", type=float, default=2e-5, help="m per frame on each free vertex (--noise rigid)")
    ap.add_argument("--val", default="", help="held-out shards (glob) for validation every 500 steps; several, "
                    "comma-separated (one per material, each logged apart; 'A|B': the first that has shards, e.g. "
                    "s1_test once it exists, else the soft pilot)")
    ap.add_argument("--noise_rel", action="store_true", help="--noise vertex: each body's noise scaled by its median "
                    "rest edge over the cloth's 1 cm (cloth unchanged; step_loss noise_rel)")
    ap.add_argument("--init", default="", help="start from this checkpoint's weights (new optimiser and schedule)")
    ap.add_argument("--K", type=int, default=8, help="chunk frames per training window (8 past frames always); --pf "
                    "can reach K (a longer pushforward, 2026-11-25)")
    ap.add_argument("--roll", type=int, default=0, help="frames of a long-rollout loss on top (lsim.train.cloth_real.rollout_loss)")
    ap.add_argument("--roll_w", type=float, default=1.0, help="its weight")
    ap.add_argument("--roll_n", type=int, default=2, help="episodes rolled out per step")
    ap.add_argument("--roll_every", type=int, default=10, help="frames between cuts of its graph")
    ap.add_argument("--roll_given", type=float, default=0.0, help="share of rollout-loss steps with the material slots given")
    ap.add_argument("--pgnd_p", type=float, default=0.0,
                    help="share of episodes drawn from the teacher on PGND's training windows (lsim.real.pgnd.load_pgnd_teacher)")
    ap.add_argument("--gen", default="", choices=("", "energy"), help="generative training: 'energy' = the energy "
                    "score of two samples on the one-step and long-rollout losses (needs --noise_dim)")
    ap.add_argument("--noise_dim", type=int, default=0, help="qwen / qwen3d: a Gaussian noise vector per body per "
                    "frame through conditional RMSNorm (lsim.model.cloth_qwen); 0: deterministic")
    ap.add_argument("--noise_lr", type=float, default=0.0, help="learning rate of the noise path's new parameters "
                    "(noise_in, the blocks' conditional-norm maps; 0: --lr): fine-tuned at 3e-5 from zero they barely grow")
    ap.add_argument("--save_every", type=int, default=2000, help="steps between saves of last.pt")
    ap.add_argument("--log_every", type=int, default=50, help="steps between log lines")
    ap.add_argument("--keep", action="store_true", help="also keep every save as stepN.pt")
    ap.add_argument("--reload", type=int, default=1000, help="steps between looking for new shards (a dataset still growing)")
    a = ap.parse_args()
    assert abs(a.fps * DT - 1.0) < 1e-9, f"--fps {a.fps} but the layer runs at {1 / DT:g} fps: set LSIM_FPS={a.fps:g}"
    assert not a.gen or a.noise_dim > 0, "--gen needs --noise_dim > 0"
    a.own = a.own or a.own_up
    import os
    from concurrent.futures import ThreadPoolExecutor
    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("RANK", 0))
    if ddp:
        torch.distributed.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dev = torch.device("cuda", torch.cuda.current_device())
    log = print if rank == 0 else (lambda *x, **k: None)
    def shard_paths():
        ps = sorted(p for d in a.data.split(",") for p in glob.glob(f"{d}/shard_*.npz")
                    if Path(p[:-4] + ".json").exists())                # a generator writes the json last
        return ps[:a.shards] if a.shards else ps
    have = set()
    paths = shard_paths()
    origin = []                                                      # each episode's shard (for --data_w)
    episodes = _load(paths, have, origin)
    if a.episodes:
        episodes, origin = episodes[:a.episodes], origin[:a.episodes]
    if a.subset:                                                     # the coverage study: a listed set of episodes
        keep = {(str(sh), int(w)) for sh, w in json.loads(Path(a.subset).read_text())["episodes"]}
        sel = [i for i, (e, o) in enumerate(zip(episodes, origin)) if (Path(o).stem, int(e.w)) in keep]
        log(f"[subset] {a.subset}: {len(sel)} of its {len(keep)} episodes found", flush=True)
        assert len(sel) >= 0.98 * len(keep), "--subset: episodes missing from --data"
        episodes, origin = [episodes[i] for i in sel], [origin[i] for i in sel]
        a.reload = 0
    log(f"{len(episodes)} cloth episodes from {len(paths)} shards", flush=True)
    dirs = [d.rstrip("/") for d in a.data.split(",")]
    weights = None
    if a.data_w:
        weights = np.array([float(x) for x in a.data_w.split(",")])
        assert len(weights) == len(dirs), "--data_w needs one weight per --data dataset"
        weights = weights / weights.sum()

    def make_groups(n_eps):                                          # episode indices per dataset (extra episodes,
        if weights is None:                                          # e.g. --pgnd_p's, go with the first)
            return None
        ds = [dirs.index(str(Path(o).parent).rstrip("/")) if o is not None else 0 for o in origin]
        ds += [0] * (n_eps - len(ds))
        ds = np.asarray(ds)
        g = [np.flatnonzero(ds == k) for k in range(len(dirs))]
        for d, gi in zip(dirs, g):
            log(f"[data] {d}: {len(gi)} episodes", flush=True)
        assert all(len(gi) for gi in g), "a --data_w dataset has no episodes"
        return g
    if a.pgnd_p > 0:                                                 # two grippers, 50-63 cm cloth: not in c1
        from lsim.real.pgnd import load_pgnd_teacher
        ext = load_pgnd_teacher()
        reps = max(1, round(a.pgnd_p * len(episodes) / ((1 - a.pgnd_p) * len(ext))))
        episodes = episodes + ext * reps
        log(f"[pgnd] {len(ext)} teacher episodes on PGND windows, x{reps}: {len(ext) * reps / len(episodes):.0%} of draws", flush=True)
    groups = make_groups(len(episodes))
    multi_val = "," in a.val or "|" in a.val                          # per material (validate_kinds)

    def make_val():                                                  # fixed held-out windows and rollout starts, once
        if multi_val:
            return make_val_kinds(val_choice(a.val), dev, log) if rank == 0 else None
        if not a.val or rank != 0 or len(glob.glob(a.val)) < 4:      # the held-out shards exist
            return None
        vrng = np.random.default_rng(12345)
        veps = [e for p in sorted(glob.glob(a.val))[:4] for e in load_shard(p[:-4])]
        vw = _windows(veps, 64, 8, 8, vrng)
        v = {"batches": [collate_cloth(vw[i:i + 4], dev, np.random.default_rng(i), p_hidden=0.0, augment=False, all_past=True)
                         for i in range(0, 64, 4)],
             "starts": [(veps[k], int(vrng.integers(8, veps[k].n_frames - 11))) for k in vrng.choice(len(veps), 16, replace=False)]}
        log(f"[val] {len(veps)} held-out episodes: 64 windows, 16 rollout starts", flush=True)
        return v
    val = make_val()
    core = build_cloth_model(vars(a), ckpt=not a.no_ckpt).to(dev)
    # the Qwen network uses every parameter every step; DDP's unused-parameter search with checkpointed, compiled
    # blocks marked parameters ready twice
    model = torch.nn.parallel.DistributedDataParallel(core, find_unused_parameters=a.arch not in ("qwen", "qwen3d")) if ddp else core
    if a.noise_lr > 0:                                               # the noise path at its own rate
        is_noise = lambda n: n.startswith("noise_in") or ".mod." in n or n.startswith("module.noise_in")
        groups_p = [[p for n, p in model.named_parameters() if not is_noise(n)], [p for n, p in model.named_parameters() if is_noise(n)]]
        opt = torch.optim.AdamW([{"params": groups_p[0], "lr": a.lr}, {"params": groups_p[1], "lr": a.noise_lr}], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.noise_lr], total_steps=a.steps, pct_start=0.05)
        log(f"[gen] noise path: {sum(p.numel() for p in groups_p[1]):,} parameters at lr {a.noise_lr:g}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start = 0
    if a.init:
        core.load_state_dict(torch.load(a.init, map_location=dev)["model"], strict=False)   # new heads start fresh
        log(f"[init] weights from {a.init}", flush=True)
    if a.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        core.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start = ck["step"] + 1
    rng = np.random.default_rng(start * 7919 + rank)
    P, K = 8, a.K
    assert a.pf <= K, f"--pf {a.pf} needs --K >= {a.pf}"
    pool = ThreadPoolExecutor(1)
    nxt = pool.submit(_windows, episodes, a.batch, P, K, np.random.default_rng(rng.integers(1 << 31)), groups, weights,
                      0.0 if a.given else None)
    t0 = time.time()
    skipped = 0
    for step in range(start, a.steps):
        wins = nxt.result()
        nxt = pool.submit(_windows, episodes, a.batch, P, K, np.random.default_rng(rng.integers(1 << 31)), groups, weights,
                      0.0 if a.given else None)
        batch = collate_cloth(wins, dev, rng, augment=not a.no_aug, all_past=(a.all_past or a.given) and not a.random_past,
                              p_hidden=0.0 if a.given else 0.5)
        opt.zero_grad(set_to_none=True)
        loss, info = step_loss(model, batch, rng, K, dev, sig=a.sig, pf=a.pf, edge_w=a.edge, noise=a.noise,
                               sig_rot=a.sig_rot, sig_white=a.sig_white, noise_rel=a.noise_rel, gen=a.gen,
                               in_jit=a.in_jit)
        if a.roll:
            from lsim.train.cloth_real import rollout_loss
            l_roll, ri = rollout_loss(core, episodes, rng, a.roll, dev, n=a.roll_n, detach_every=a.roll_every, p_given=a.roll_given,
                                      gen=a.gen)
            loss = loss + a.roll_w * l_roll
            info.update(ri)
        loss.backward()
        if a.side_apart:
            side = lambda n: n.removeprefix("module.").startswith(("unc.", "param."))
            torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if side(n)], 1.0)
            gn = torch.nn.utils.clip_grad_norm_([p for n, p in model.named_parameters() if not side(n)], 1.0)
        else:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isfinite(gn):                                         # one non-finite gradient (a rollout through
            opt.step()                                                 # the shape restoration) once turned every
        else:                                                          # weight NaN by step 50 (2026-11-05): skip it
            skipped += 1
            opt.zero_grad(set_to_none=True)
        info["skipped"] = skipped
        sched.step()
        if step % a.log_every == 0:
            log(f"step {step:6d}  loss {loss.item():.3e}  " + "  ".join(f"{k} {v:.3g}" for k, v in info.items()) +
                f"  grad {gn.item():.2g}  lr {sched.get_last_lr()[0]:.2g}  tokens {batch['pos'].shape[0] * batch['pos'].shape[1]}"
                f"  {time.time() - t0:.0f} s", flush=True)
        if a.reload and step % a.reload == 0 and step > start:
            new_origin = []
            new = _load(shard_paths(), have, new_origin)
            if new:
                n_old = len(episodes)
                episodes = episodes + new
                if groups is not None:                                   # new shards' episodes into their datasets
                    ds = np.asarray([dirs.index(str(Path(o).parent).rstrip("/")) for o in new_origin])
                    groups = [np.concatenate([g, n_old + np.flatnonzero(ds == k)]) for k, g in enumerate(groups)]
                log(f"[data] {len(episodes)} episodes from {len(have)} shards", flush=True)
            if val is None or (multi_val and rank == 0 and val_choice(a.val) != val["choice"]):
                val = make_val()                                         # (a preferred held-out set appeared)
        if val is not None and multi_val and (step % 500 == 0 or step == a.steps - 1):
            validate_kinds(model, val, dev, step, log)
        elif val is not None and (step % 500 == 0 or step == a.steps - 1):
            one, roll = validate(model, val, dev)
            log(f"[val] step {step}  held-out clean step {one:.3f} mm  10-frame rollout {roll:.2f} mm (median)", flush=True)
        if rank == 0 and (step % a.save_every == 0 or step == a.steps - 1):
            torch.save({"model": core.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "args": vars(a), "step": step}, out / "last.pt")
            if a.keep and step > start:
                torch.save({"model": core.state_dict(), "args": vars(a), "step": step}, out / f"step{step}.pt")
    if ddp:
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
