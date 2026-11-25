"""Per-frame contact bookkeeping on the GPU (PHASE1_AMENDMENT.md section 6).

Every substep, each solver contact adds its force to a per-world matrix F[w, a, b] = force on
slot a exerted by slot b (world frame), its torque about slot a's centre of mass to T[w, a, b]
(exact: a force-weighted mean contact point loses the couple, and replaying force + mean point
missed rotating objects by 44 mm over 8 frames), and its world contact point, weighted by force
magnitude, to P[w, a, b] (so P / Wt is where the force acts). Slots per world: 0 = static
geometry (table, cabinet), 1 = gripper, 2.. = objects, then hinged links. Dividing by the
substep count gives frame-averaged values (impulse / frame time).
"""
import numpy as np
import warp as wp


@wp.kernel
def _accumulate(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32),
                point0: wp.array(dtype=wp.vec3), contact_f: wp.array(dtype=wp.spatial_vector),
                shape_body: wp.array(dtype=wp.int32), shape_world: wp.array(dtype=wp.int32),
                body_slot: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
                body_com: wp.array(dtype=wp.vec3), K: int,
                F: wp.array(dtype=wp.vec3), P: wp.array(dtype=wp.vec3), Wt: wp.array(dtype=wp.float32),
                T: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if i >= count[0]:
        return
    s0 = shape0[i]
    s1 = shape1[i]
    if s0 < 0 or s1 < 0:
        return
    b0 = shape_body[s0]
    b1 = shape_body[s1]
    a = 0
    b = 0
    if b0 >= 0:
        a = body_slot[b0]
    if b1 >= 0:
        b = body_slot[b1]
    if a < 0 or b < 0:
        return
    w = shape_world[s0]
    if w < 0:
        w = shape_world[s1]
    if w < 0:
        return
    f = wp.spatial_top(contact_f[i])                        # force on body0 by body1
    p = point0[i]
    if b0 >= 0:
        p = wp.transform_point(body_q[b0], point0[i])
    m = wp.length(f)
    ab = (w * K + a) * K + b
    ba = (w * K + b) * K + a
    wp.atomic_add(F, ab, f)
    wp.atomic_add(F, ba, -f)
    wp.atomic_add(P, ab, m * p)
    wp.atomic_add(P, ba, m * p)
    wp.atomic_add(Wt, ab, m)
    wp.atomic_add(Wt, ba, m)
    if b0 >= 0:                                             # torque on body0 about its CoM
        wp.atomic_add(T, ab, wp.cross(p - wp.transform_point(body_q[b0], body_com[b0]), f))
    if b1 >= 0:
        wp.atomic_add(T, ba, wp.cross(p - wp.transform_point(body_q[b1], body_com[b1]), -f))


class ContactRecorder:
    def __init__(self, teacher, body_slot, n_worlds, K):
        self.t, self.W, self.K = teacher, n_worlds, K
        dev = teacher.model.device
        self.body_slot = wp.array(np.asarray(body_slot, np.int32), dtype=wp.int32, device=dev)
        n = n_worlds * K * K
        self.F = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.P = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.Wt = wp.zeros(n, dtype=wp.float32, device=dev)
        self.T = wp.zeros(n, dtype=wp.vec3, device=dev)
        self.substeps = 0

    def accumulate(self):
        c, m = self.t.contacts, self.t.model
        wp.launch(_accumulate, dim=c.rigid_contact_max,
                  inputs=[c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                          self.t.obs.contact_f, m.shape_body, m.shape_world, self.body_slot, self.t.s0.body_q,
                          m.body_com, self.K],
                  outputs=[self.F, self.P, self.Wt, self.T], device=m.device)
        self.substeps += 1

    def start(self, n_frames):
        """Keep every frame on the device (one transfer at the end instead of four per frame:
        the per-frame copies made episode generation sync-bound)."""
        n = self.W * self.K * self.K
        dev = self.t.model.device
        self.rec = {k: wp.zeros(n_frames * n, dtype=dt, device=dev)
                    for k, dt in (("F", wp.vec3), ("P", wp.vec3), ("Wt", wp.float32), ("T", wp.vec3))}
        self.n_frames = n_frames
        self.reset()

    def reset(self):
        self.F.zero_(); self.P.zero_(); self.Wt.zero_(); self.T.zero_()
        self.substeps = 0

    def store(self, f):
        """Move the current frame's sums into slot f (device to device) and reset."""
        n = self.W * self.K * self.K
        for k in ("F", "P", "Wt", "T"):
            wp.copy(self.rec[k], getattr(self, k), dest_offset=f * n, count=n)
        self.per_frame = max(self.substeps, 1)
        self.reset()

    def collect(self):
        """All stored frames: frame-averaged force, force-weighted point, torque, (nf, W, K, K, 3)."""
        shape = (self.n_frames, self.W, self.K, self.K)
        F = self.rec["F"].numpy().reshape(shape + (3,)) / self.per_frame
        T = self.rec["T"].numpy().reshape(shape + (3,)) / self.per_frame
        wt = self.rec["Wt"].numpy().reshape(shape + (1,))
        P = np.where(wt > 0, self.rec["P"].numpy().reshape(shape + (3,)) / np.maximum(wt, 1e-12), np.nan)
        return F.astype(np.float32), P.astype(np.float32), T.astype(np.float32)

    def frame(self):
        """Frame-averaged force, force-weighted contact point and torque about slot a's CoM,
        each (W, K, K, 3); resets."""
        n = max(self.substeps, 1)
        F = self.F.numpy().reshape(self.W, self.K, self.K, 3) / n
        T = self.T.numpy().reshape(self.W, self.K, self.K, 3) / n
        wt = self.Wt.numpy().reshape(self.W, self.K, self.K, 1)
        P = np.where(wt > 0, self.P.numpy().reshape(self.W, self.K, self.K, 3) / np.maximum(wt, 1e-12), np.nan)
        self.F.zero_(); self.P.zero_(); self.Wt.zero_(); self.T.zero_()
        self.substeps = 0
        return F.astype(np.float32), P.astype(np.float32), T.astype(np.float32)
