"""Frame-by-frame rollouts (inference). The network sees the last P frames and predicts the
forces for the next step; the physics layer moves every dynamic rigid body from its own state;
hinged parts move along their joint; the gripper and static surfaces follow the recording. Hidden parameter slots are
predicted once from the true past and carried through the rollout (DIRECTION.md "Time")."""
import copy

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from lsim.train.train import collate, damping_slot, physics_step, slots
from lsim.train.windows import F_EMPTY, F_GIVEN, make_window


def rigid_fit(local, world):
    """Pose (x, y, z, qx, qy, qz, qw) best mapping body-frame points onto world points (Kabsch)."""
    lc, wc = local.mean(0), world.mean(0)
    U, _, Vt = np.linalg.svd((local - lc).T @ (world - wc))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return np.r_[wc - R @ lc, Rotation.from_matrix(R).as_quat()].astype(np.float32)


@torch.no_grad()
def rollout(model, ep, t0, H, P=8, hidden=(), grip_hidden=False, dev="cuda", seed=0):
    """Predict frames t0+1 .. t0+H from the recording up to t0. Returns (pred, true): world poses
    (H, n_bodies, 7) of the episode's bodies, in ep.bodies order, and the slot predictions."""
    rng = np.random.default_rng(seed)
    view = copy.copy(ep)
    view.q = ep.q.copy()                                       # predicted poses are written into it
    ids = [b.body for b in ep.bodies]
    dyn = np.array([b.role == "dynamic" for b in ep.bodies])          # rigid objects and hinged parts
    carried = None
    for t in range(t0, t0 + H):
        w = make_window(view, t, P, 0, rng, near=getattr(model, "near", 0.03),          # frames up to now only
                        p_hide_grip=1.0 if grip_hidden else 0.0)
        b = collate([w], torch.device(dev), rng, p_hist_noise=0.0, augment=False, all_past=True, hidden=hidden)
        Fw = b["pos"].shape[1]
        b_in = dict(b, tau=torch.zeros(Fw, device=dev), frame_given=torch.ones(Fw, device=dev))
        f_contact, f_ext_pred, pred = model(b_in, lambda pr: slots(b, pr if carried is None else carried)[0])
        if carried is None:
            carried = pred
        m, c, J = slots(b, carried)
        st = b["f_state"][:, P - 1]
        f_ext = torch.where((st == F_GIVEN)[:, None], b["f_ext"][:, P - 1], torch.zeros_like(b["f_ext"][:, P - 1]))
        f_ext = torch.where((st == F_EMPTY)[:, None], f_ext_pred[:, P - 1], f_ext)
        if getattr(model, "direct", False):                               # B2: vertices, then a rigid fit per body
            x = (2 * b["pos"][:, P - 1] - b["pos"][:, P - 2] + f_contact[:, P - 1] * (1 / 30.0) ** 2).cpu().numpy()
            bo, vl = b["body_of"].cpu().numpy(), b["vert_local"].cpu().numpy()
            for i, bi in enumerate(ids):
                if dyn[i]:
                    view.q[t + 1, bi] = rigid_fit(vl[bo == i], x[bo == i])
            continue
        new = physics_step(b, b["body_q_now"], b["body_q_prev"], f_contact[:, P - 1] + f_ext, b["pos"][:, P - 1],
                           m, c, J, damping_slot(b, carried)).cpu().numpy()
        for i, bi in enumerate(ids):
            if dyn[i]:
                view.q[t + 1, bi] = new[i]
    return view.q[t0 + 1:t0 + H + 1][:, ids], ep.q[t0 + 1:t0 + H + 1][:, ids], carried
