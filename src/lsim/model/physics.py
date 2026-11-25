"""Exact physics layer (DIRECTION.md "Network", step 4): no learned weights.

Per-vertex forces (learned contact forces + given external forces) are summed into each rigid
body's force and torque about its centre of mass, gravity is added, and the body is integrated
with its mass and inertia (given or inferred) by semi-implicit Euler, one step per frame. Hinged
links keep only the motion along their joint (prismatic: translation along the axis; revolute:
rotation about the axis through the pivot), with joint damping applied exactly and limits as
inelastic stops. Vertices then follow their body.

All tensors are batched over bodies; quaternions are (x, y, z, w) like Newton's.
"""
import torch

G = torch.tensor([0.0, 0.0, -9.81])


def qmul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qrot(q, v):
    """Rotate v (..., 3) by unit quaternion q (..., 4)."""
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def qexp(w, dt):
    """Quaternion for rotating by angular velocity w (..., 3) over dt."""
    ang = torch.linalg.norm(w, dim=-1, keepdim=True) * dt
    axis = w / torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.cat([axis * torch.sin(ang / 2), torch.cos(ang / 2)], -1)


V_MAX, W_MAX = 20.0, 300.0                       # m/s, rad/s: nothing on a tabletop moves faster


def _skew(a):
    z = torch.zeros_like(a[..., 0])
    return torch.stack([z, -a[..., 2], a[..., 1], a[..., 2], z, -a[..., 0], -a[..., 1], a[..., 0], z],
                       -1).reshape(a.shape[:-1] + (3, 3))


def rigid_step(pos, quat, v, w, force, torque, mass, inertia_body, dt, gravity=True, clamp=True):
    """One semi-implicit Euler step for rigid bodies; the gyroscopic term is implicit.

    pos, v, w, force, torque: (B, 3) world frame, pos = centre of mass; quat (B, 4) body->world;
    mass (B,); inertia_body (B, 3, 3) about the CoM in the body frame.

    Euler's equations in the body frame, I (w1 - w0) + dt w1 x I w1 = dt tau, solved with one
    Newton step from w0 (Catto, GDC 2015). The explicit term w0 x I w0 diverged at 30 Hz for an
    anisotropic body spinning at ~30 rad/s (w dt 1.1 -> 79 -> 5781 in three steps).
    """
    a = force / mass[:, None] + (G.to(pos) if gravity else 0.0)
    v1 = v + a * dt
    R = _qmat(quat)
    wb = (R.transpose(-1, -2) @ w[..., None])[..., 0]
    tb = (R.transpose(-1, -2) @ torque[..., None])[..., 0]
    Iwb = (inertia_body @ wb[..., None])[..., 0]
    f = dt * (torch.cross(wb, Iwb, dim=-1) - tb)
    J = inertia_body + dt * (_skew(wb) @ inertia_body - _skew(Iwb))
    wb1 = wb - torch.linalg.solve(J, f[..., None])[..., 0]
    w1 = (R @ wb1[..., None])[..., 0]
    if clamp:                                        # a guard for untrained forces, not physics
        v1 = v1 * (V_MAX / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp_min(V_MAX))
        w1 = w1 * (W_MAX / torch.linalg.norm(w1, dim=-1, keepdim=True).clamp_min(W_MAX))
    pos1 = pos + v1 * dt
    q1 = qmul(qexp(w1, dt), quat)
    return pos1, q1 / torch.linalg.norm(q1, dim=-1, keepdim=True), v1, w1


def _qmat(q):
    x, y, z, w = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def wrench(f_vert, x_vert, body_of_vertex, com, n_bodies):
    """Sum per-vertex forces (V, 3) at positions (V, 3) into per-body force and torque about com."""
    F = torch.zeros(n_bodies, 3, dtype=f_vert.dtype, device=f_vert.device).index_add_(0, body_of_vertex, f_vert)
    r = x_vert - com[body_of_vertex]
    T = torch.zeros(n_bodies, 3, dtype=f_vert.dtype, device=f_vert.device).index_add_(
        0, body_of_vertex, torch.cross(r, f_vert, dim=-1))
    return F, T


def prismatic_step(q, qd, f_axis, mass, damping, dt, lo, hi):
    """Joint coordinate update along a prismatic axis; damping exact; limits inelastic."""
    qdd = (f_axis - damping * qd) / mass
    qd1 = qd + qdd * dt
    q1 = q + qd1 * dt
    hit = (q1 < lo) | (q1 > hi)
    return q1.clamp(lo, hi), torch.where(hit, torch.zeros_like(qd1), qd1)


def revolute_step(q, qd, tau_axis, inertia_axis, damping, dt, lo, hi):
    qdd = (tau_axis - damping * qd) / inertia_axis
    qd1 = qd + qdd * dt
    q1 = q + qd1 * dt
    hit = (q1 < lo) | (q1 > hi)
    return q1.clamp(lo, hi), torch.where(hit, torch.zeros_like(qd1), qd1)


def joint_coord(kind, axis, pivot, q):
    """Joint coordinate of hinged bodies from origin poses q (B, 7): the travel along the axis
    from the pivot (prismatic, kind 1) or the angle about the axis from the identity (kind 2)."""
    s = ((q[:, :3] - pivot) * axis).sum(-1)
    th = 2.0 * torch.atan2((q[:, 3:6] * axis).sum(-1), q[:, 6])
    return torch.where(kind == 1, s, th)


def joint_pose(kind, axis, pivot, q, dq):
    """Origin poses q (B, 7) moved by dq along each body's joint."""
    p_pr = q[:, :3] + dq[:, None] * axis
    rot = torch.cat([axis * torch.sin(dq / 2)[:, None], torch.cos(dq / 2)[:, None]], -1)
    p_rv = pivot + qrot(rot, q[:, :3] - pivot)
    q_rv = qmul(rot, q[:, 3:])
    pr = (kind == 1)[:, None]
    return torch.cat([torch.where(pr, p_pr, p_rv), torch.where(pr, q[:, 3:], q_rv)], -1)


def joint_step(kind, axis, pivot, lim, q_prev, q_cur, F, T_pivot, mass, com_local, inertia_body, damping, dt):
    """One step of every hinged body along its joint, from two poses (frame-average velocity).

    F, T_pivot: (B, 3) total force and torque about the pivot from the vertices; gravity is added
    here. Damping is implicit (stable at any c dt / M); limits are inelastic stops. Returns the
    new origin poses (B, 7)."""
    c_prev, c_cur = joint_coord(kind, axis, pivot, q_prev), joint_coord(kind, axis, pivot, q_cur)
    qd = (c_cur - c_prev) / dt
    Fg = G.to(F) * mass[:, None]
    R = _qmat(q_cur[:, 3:])
    r = q_cur[:, :3] + qrot(q_cur[:, 3:], com_local) - pivot
    Q = torch.where(kind == 1, ((F + Fg) * axis).sum(-1), ((T_pivot + torch.cross(r, Fg, dim=-1)) * axis).sum(-1))
    Iw = R @ inertia_body @ R.transpose(-1, -2)
    I_ax = (axis * (Iw @ axis[..., None])[..., 0]).sum(-1) + mass * torch.cross(r, axis, dim=-1).pow(2).sum(-1)
    M = torch.where(kind == 1, mass, I_ax).clamp_min(1e-8)
    qd1 = (qd + Q / M * dt) / (1.0 + damping / M * dt)
    c1 = c_cur + qd1 * dt
    lo, hi = lim[:, 0], lim[:, 1]
    c1c = torch.minimum(torch.maximum(c1, lo), hi)
    return joint_pose(kind, axis, pivot, q_cur, c1c - c_cur)
