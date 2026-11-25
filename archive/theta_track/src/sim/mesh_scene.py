"""Rigid-body scene whose contact geometry IS the mesh (no sphere cover).

Drop-in for ProbeScene: same constructor arguments, same positions()/rotations() output,
same per-body mu/cd/mass and table material, same gravity_seq ramp. Only the collision
representation changes.
"""
import numpy as np
import warp as wp

from ..data.assets import decimate, load_asset
from .diff_collide_6dof import integrate_6dof
from .mesh_contact import (apply_links, apply_revolute, ground_contact_mesh,
                           pair_contact_mesh, vertex_weights, world_verts)


def mass_points(tm, model, n_solid=60):
    """(points, weights summing to 1) describing where a body's mass is.

    shell / vertex: the mesh vertices with their area weights (vertex_weights) -- the
    distribution the contact forces act on. solid: centres of interior voxels, found by
    sampling the surface densely and flood-filling the exterior (numpy only; works for open
    scans, whose cavities then count as outside)."""
    V = np.asarray(tm.vertices, np.float64)
    if model in ("shell", "vertex"):
        w = vertex_weights(tm).astype(np.float64)
        return V, w / w.sum()
    if model != "solid":
        raise ValueError(f"mass_model must be shell / solid / vertex, got {model!r}")
    from collections import deque
    pitch = float(np.ptp(V, 0).max()) / n_solid
    import trimesh
    pts = np.asarray(trimesh.sample.sample_surface(tm, 300000, seed=0)[0])  # seeded: deterministic
    # Grid offset by an IRRATIONAL fraction of a voxel. A flat face lying exactly on a voxel
    # boundary rounds into two partial layers whose gaps the 6-connected fill leaks through
    # -- a solid cube then came out with its SHELL inertia. Thickening the wall to close the
    # gaps grew the body outward instead (+10% inertia, +3% centre of mass on a hemisphere).
    lo = V.min(0) - pitch * (1.0 + 0.3819660)
    ijk = np.floor((pts - lo) / pitch).astype(int)
    S = np.zeros(ijk.max(0) + 3, bool); S[tuple(ijk.T)] = True
    out = np.zeros_like(S); out[0, 0, 0] = True; dq = deque([(0, 0, 0)])
    while dq:
        x, y, z = dq.popleft()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = x + dx, y + dy, z + dz
            if 0 <= a < S.shape[0] and 0 <= b < S.shape[1] and 0 <= c < S.shape[2] \
                    and not out[a, b, c] and not S[a, b, c]:
                out[a, b, c] = True; dq.append((a, b, c))
    solid = ~out
    # boundary (wall) voxels straddle the true surface -- about half inside on average;
    # weighting them 0.5 removes the outward bias of counting them whole
    idx = np.argwhere(solid)
    w = np.where(S[tuple(idx.T)], 0.5, 1.0)
    P = lo + (idx + 0.5) * pitch
    return P, w / w.sum()


class MeshProbeScene:
    def __init__(self, names, pos0, vel0, ang0=None, rot0=None, densities=None, masses=None,
                 ground_z=0.706, dt=1.0 / 1440.0, n_steps=1400, k=2500.0, cd=3000.0,
                 mu=0.5, gravity=(0.0, 0.0, -9.81), mesh_scale=None, faces=1200,
                 ground_mu=None, ground_cd=None, query_dist=0.05, table=None,
                 com_offset=None, tilt_deg=0.0, pivot=None, mass_model="auto"):
        self.N = len(names)
        # MASS MODEL -- where a body's mass is, per body (str or list). It sets the centre of
        # mass AND the inertia from one point cloud, so the two can no longer disagree:
        #   "shell"   area-weighted surface (thin-walled hollow objects)
        #   "solid"   voxel-filled interior (solid objects)
        #   "vertex"  LEGACY: centre of mass = plain vertex mean, inertia = area-weighted
        #             cloud about it. Inconsistent with itself -- vertex density follows
        #             modelling detail, not mass -- and off by up to 2.2x in base leverage b/h
        #             (scripts/f14_llm_outcomes.py). Kept only to reproduce pre-2026-10-05 runs.
        #   "auto"    per asset from src/data/assets.MASS_MODEL (the DEFAULT); procedural
        #             trimeshes passed directly get "shell"
        from ..data.assets import mass_model_for
        mm = [mass_model] * self.N if isinstance(mass_model, str) else list(mass_model)
        self.mass_model = [mass_model_for(n) if m == "auto" else m for m, n in zip(mm, names)]
        self._mass_pts, self._mass_w = [], []
        self.dt, self.n_steps = float(dt), int(n_steps)
        self.ground_z = float(ground_z)
        # A REAL TILTED PLANE, rather than a level plane with gravity rotated. tilt_deg
        # rotates the ground about +Y through `pivot` (default: the table centre at
        # ground_z), so +x runs downhill -- matching how blender_render_scene rotates the
        # table. gravity stays (0,0,-9.81), the simulator and the renderer share one frame,
        # and nothing downstream has to transform poses by hand.
        self.tilt_deg = float(tilt_deg)
        th = np.radians(self.tilt_deg)
        self.pivot = (np.asarray(pivot, np.float64) if pivot is not None
                      else np.array([0.0, 0.0, self.ground_z]))
        # Ry(th) applied to +z and +x. +x tilts downward, so the surface falls toward +x.
        self._gn = np.array([np.sin(th), 0.0, np.cos(th)])
        self._gt = np.array([np.cos(th), 0.0, -np.sin(th)])
        self._gp = self.pivot.copy()
        self.gravity = wp.vec3(*gravity)
        self.gravity_seq = None
        self._gmu = float(ground_mu) if ground_mu is not None else -1.0
        self._gcd = float(ground_cd) if ground_cd is not None else -1.0
        self.query_dist = float(query_dist)
        # (half_width, half_depth) of the table, or None for an unbounded plane
        self.half_w, self.half_d = (float(table[0]), float(table[1])) if table else (-1.0, -1.0)

        # com_offset[bi] moves a body's CENTRE OF MASS off its geometric centre, in metres
        # in the asset's own axes. It is applied by re-centring the local vertex cloud: the
        # body frame origin IS the centre of mass everywhere downstream -- gravity acts
        # there, inertia is taken about it, and contacts are expressed in it -- so shifting
        # the origin is the whole change and the solver needs none. Renders must undo it,
        # which is the COM-offset convention balance_render_prep.py already follows.
        self.com_shift = [np.zeros(3) for _ in range(self.N)]
        loc, bod, wts, self.meshes, self.coms, self.sizes, vols = [], [], [], [], [], [], []
        mesh_ids = []
        for bi, name in enumerate(names):
            if hasattr(name, "vertices"):
                # a trimesh passed directly -- procedural rig parts (balance beam, pans)
                # live in the same scene as scanned assets and need the same exact contact
                tm = name.copy()
            else:
                cat, _, nm = name.rpartition("/")
                tm = decimate(load_asset(cat or "rigid", nm), faces).copy()
            if mesh_scale is not None:
                s = float(mesh_scale[bi] if hasattr(mesh_scale, "__len__") else mesh_scale)
                tm.apply_scale(s)
            V = np.asarray(tm.vertices, np.float64)
            mpts, mw = mass_points(tm, self.mass_model[bi])
            C = V.mean(0) if self.mass_model[bi] == "vertex" else (mpts * mw[:, None]).sum(0)
            if com_offset is not None and com_offset[bi] is not None:
                d = np.asarray(com_offset[bi], np.float64).reshape(3)
                self.com_shift[bi] = d
                C = C + d
            # Body frame is COM-centred, matching how poses are exported. Unlike the sphere
            # cover this is the ONLY offset -- the collision surface and the visible surface
            # are now the same object, so nothing floats.
            Vc = (V - C).astype(np.float32)
            self._mass_pts.append(mpts - C); self._mass_w.append(mw)
            self.coms.append(C)
            self.sizes.append(float(max(tm.extents)))
            vols.append(float(abs(tm.volume)) if tm.is_watertight else float(np.prod(tm.extents)) * 0.5)
            loc.append(Vc)
            bod.append(np.full(len(Vc), bi, np.int32))
            wts.append(vertex_weights(tm).astype(np.float32))
            m = wp.Mesh(points=wp.array(Vc, dtype=wp.vec3),
                        indices=wp.array(np.asarray(tm.faces, np.int32).flatten(), dtype=int))
            self.meshes.append(m); mesh_ids.append(m.id)

        self.local = wp.array(np.concatenate(loc), dtype=wp.vec3)
        self.body = wp.array(np.concatenate(bod), dtype=int)
        self.weight = wp.array(np.concatenate(wts), dtype=float)
        self.mesh_ids = wp.array(np.array(mesh_ids, np.uint64), dtype=wp.uint64)
        self.nV = len(self.body)
        self.volumes = np.asarray(vols)

        self.mass = wp.zeros(self.N, dtype=float)
        self.inv_mass = wp.zeros(self.N, dtype=float)
        self.G = wp.zeros(self.N, dtype=wp.mat33)
        self.Ginv = wp.zeros(self.N, dtype=wp.mat33)
        self._loc_np, self._bod_np, self._w_np = (np.concatenate(loc), np.concatenate(bod),
                                                  np.concatenate(wts))
        self.set_masses(masses if masses is not None
                        else np.asarray(densities) * self.volumes)

        arr = lambda v: wp.array(np.full(self.N, float(v), np.float32) if np.isscalar(v)
                                 else np.asarray(v, np.float32).reshape(self.N), dtype=float)
        self.mu, self.cd, self.k = arr(mu), arr(cd), arr(k)

        self.pos0 = np.asarray(pos0, np.float32)
        self.vlin0 = np.asarray(vel0, np.float32)
        self.vang0 = np.asarray(ang0 if ang0 is not None else np.zeros((self.N, 3)), np.float32)
        # INITIAL ORIENTATION. rot[0] was hard-coded to identity, so a body could only ever
        # start in its authored pose -- which makes a rocking probe impossible, since a bowl
        # sitting flat does not rock. Supplying rot0 lets the excitation be a controlled
        # DISPLACEMENT (tip it and release) rather than a velocity kick, and a displacement
        # is what makes the initial amplitude the same for every candidate.
        self.rot0 = (np.tile([0.0, 0.0, 0.0, 1.0], (self.N, 1)).astype(np.float32)
                     if rot0 is None else np.asarray(rot0, np.float32).reshape(self.N, 4))
        mk = lambda d: [wp.zeros(self.N, dtype=d) for _ in range(self.n_steps + 1)]
        self.pos, self.rot = mk(wp.vec3), mk(wp.quat)
        self.vlin, self.vang = mk(wp.vec3), mk(wp.vec3)
        self.force = wp.zeros(self.N, dtype=wp.vec3)
        self.torque = wp.zeros(self.N, dtype=wp.vec3)
        self.world = wp.zeros(self.nV, dtype=wp.vec3)
        # revolute joints: off for every body unless set_hinge() is called
        self.is_hinge = wp.zeros(self.N, dtype=int)
        self.anchor = wp.zeros(self.N, dtype=wp.vec3)
        self.axis = wp.array(np.tile([0.0, 1.0, 0.0], (self.N, 1)).astype(np.float32),
                             dtype=wp.vec3)
        self._hinge_damp = wp.array(np.ones(self.N, np.float32), dtype=float)
        self._hinge_limit = wp.zeros(self.N, dtype=float)     # 0 = unlimited swing
        self._pivot_local = wp.zeros(self.N, dtype=wp.vec3)   # pivot offset in body frame
        self._links = []                                      # (a, b, local_a, local_b, k, c)
        self._link_arrays = None

    def set_masses(self, m):
        m = np.asarray(m, np.float64).reshape(self.N)
        self.mass.assign(m.astype(np.float32))
        self.inv_mass.assign((1.0 / m).astype(np.float32))
        # Inertia from the area-weighted vertex cloud -- the same distribution the contact
        # forces act on, so torque and inertia stay consistent even for open meshes.
        Gs, Gis = [], []
        for b in range(self.N):
            P, w = self._mass_pts[b], self._mass_w[b]
            r2 = (P ** 2 * w[:, None]).sum(0)
            G = np.diag([r2[1] + r2[2], r2[0] + r2[2], r2[0] + r2[1]])
            off = np.zeros((3, 3))
            for a in range(3):
                for c in range(3):
                    if a != c:
                        off[a, c] = -(P[:, a] * P[:, c] * w).sum()
            # PER-UNIT-MASS inertia. integrate_6dof forms the world inertia as
            #     Iw = R * (mass * G) * R^T
            # so G must NOT already carry the mass. Multiplying it in here made the inertia
            # m^2*G_unit -- off by a factor of m, which for a 0.1 kg body is 10x too SMALL
            # and gives ten times the angular acceleration it should. Bodies under a torque
            # then wind up instead of settling.
            G = G + off
            G += np.eye(3) * (1e-9 + 1e-6 * np.trace(G) / 3.0)   # keep it invertible
            Gs.append(G.astype(np.float32)); Gis.append(np.linalg.inv(G).astype(np.float32))
        self.G.assign(np.stack(Gs)); self.Ginv.assign(np.stack(Gis))

    def set_hinge(self, bi, anchor, axis=(0.0, 1.0, 0.0), damp_per_sec=6.0,
                  limit_deg=None, pivot_local=None):
        """Pin body `bi` to `anchor`, free to rotate only about `axis`. Exact, no stiffness.

        `damp_per_sec` is the exponential decay rate of angular velocity, converted to a
        per-step factor -- so settling time is a physical time, independent of dt. It is
        stored PER BODY: a scene can hold a balance beam that swings and settles alongside a
        stand that must never move at all, and a single shared value would let whichever
        hinge was configured last silently overwrite the other.
        """
        if pivot_local is not None:
            V = self._pivot_local.numpy()
            V[bi] = np.asarray(pivot_local, np.float32); self._pivot_local.assign(V)
        if limit_deg is not None:
            L = self._hinge_limit.numpy()
            L[bi] = float(np.radians(limit_deg)); self._hinge_limit.assign(L)
        d = self._hinge_damp.numpy()
        d[bi] = float(np.exp(-float(damp_per_sec) * self.dt))
        self._hinge_damp.assign(d)
        h = self.is_hinge.numpy(); h[bi] = 1; self.is_hinge.assign(h)
        a = self.anchor.numpy(); a[bi] = np.asarray(anchor, np.float32); self.anchor.assign(a)
        x = self.axis.numpy(); x[bi] = np.asarray(axis, np.float32); self.axis.assign(x)

    def _set_hinge_arrays(self, bi, anchor, axis=(0.0, 1.0, 0.0)):
        """Deprecated alias kept for callers predating per-body hinge damping."""
        h = self.is_hinge.numpy(); h[bi] = 1; self.is_hinge.assign(h)
        a = self.anchor.numpy(); a[bi] = np.asarray(anchor, np.float32); self.anchor.assign(a)
        x = self.axis.numpy(); x[bi] = np.asarray(axis, np.float32); self.axis.assign(x)

    def add_link(self, a, b, local_a, local_b, stiffness=4.0e4, damping=40.0):
        """Suspend body `a` from body `b` by a stiff point-to-point spring.

        `local_a` / `local_b` are attachment points in each body's own COM-centred frame. Put
        the pan's point above its centre of mass and it hangs level without any constraint on
        its orientation.
        """
        self._links.append((int(a), int(b), tuple(local_a), tuple(local_b),
                            float(stiffness), float(damping)))
        L = self._links
        self._link_arrays = (
            wp.array(np.array([x[0] for x in L], np.int32), dtype=int),
            wp.array(np.array([x[1] for x in L], np.int32), dtype=int),
            wp.array(np.array([x[2] for x in L], np.float32), dtype=wp.vec3),
            wp.array(np.array([x[3] for x in L], np.float32), dtype=wp.vec3),
            wp.array(np.array([x[4] for x in L], np.float32), dtype=float),
            wp.array(np.array([x[5] for x in L], np.float32), dtype=float))
        return len(L) - 1

    def plane_quat(self):
        """Body orientation that puts the body's +z along the plane normal, as (x,y,z,w)."""
        h = np.radians(self.tilt_deg) / 2.0
        return [0.0, float(np.sin(h)), 0.0, float(np.cos(h))]

    def seat_on_plane(self, bi=0, along=0.0, across=0.0, clearance=0.0, rotate=True):
        """Place body `bi` on the tilted plane, `along` metres downhill of the pivot.

        Two things have to happen together and neither is obvious, which is why they live
        here rather than in each script:

        ROTATE THE BODY TO THE PLANE. With gravity rotated and the ground level, a body
        with identity orientation was automatically flat on the ramp. On a REAL tilted
        plane the same body starts tipped by the tilt angle, so it rests on the edge of its
        base instead of its face: only a sliver of weighted contact area engages, the
        effective stiffness collapses, and a 0.4 kg bowl sank 14.5 mm and was ejected,
        sliding 4 m at a friction 2.2x above its slip threshold. Rotating it by the same
        Ry(tilt) fixes that.

        SEAT ALONG THE NORMAL, not along z. The lowest point of the body must be measured
        against the plane. Because the body carries the plane's own rotation, the lowest
        vertex along the normal is simply min(local z) -- but only once `rotate` has been
        applied, so the two steps are not separable.

        CLEARANCE IS 0, NOT 2 mm. The old default dropped the body 2 mm onto the plane at
        release; on a 20 deg slope that impact gave it a ~6 cm/s head start along the
        ramp, an 11% overshoot at 1 s that read as a friction error until the acceleration
        was checked (exact to 1%). Found by fitting Newton to real Physics 101 footage
        (scripts/p4_sim_loop.py). Under calibrate_stiffness the body sinks only
        pen_frac * size (~20 um) under its own weight, so starting in contact is benign.
        Results computed before 2026-10-05 (o13, o14, f4/f6, f7, p4) carry the 2 mm drop.
        """
        q = self.plane_quat()
        if rotate:
            R = np.asarray(self.rot0, np.float32).copy()
            R[bi] = np.asarray(q, np.float32)
            self.rot0 = R
        L = self.local.numpy().astype(float)
        L = L[self._bod_np == bi]
        low = float(L[:, 2].min())
        P = np.asarray(self.pos0, np.float32).copy()
        P[bi] = (self._gp + self._gt * float(along)
                 + np.cross(self._gn, self._gt) * float(across)
                 + self._gn * (-low + float(clearance)))
        self.pos0 = P
        return self

    def set_restitution_alpha(self, alpha):
        """Contact damping from a Hunt-Crossley alpha (s/m) instead of a raw cd.

        The contact force is fn = k w pen - HC cd w pen vn (Hunt-Crossley), whose
        restitution is ~ 1 - alpha * v_impact with alpha = 2 HC cd / (3 k). So restitution
        is set by cd / k, not by cd: on the tilt rig (pen_frac 1e-4, k 50x stiffer) every
        cd up to 30000 gave e = 0.77-1.00, and on any rig one cd gave bodies of different
        mass different restitution (k is set proportional to mass). Specifying alpha and
        deriving cd = 1.5 alpha k / HC per body makes restitution independent of stiffness,
        mass and rig (scripts/p8_alpha.py). Re-applied by calibrate_stiffness, so the order
        of the two calls does not matter. Turns off the ground_cd geometric-mean mix, which
        would otherwise reintroduce a fixed, unscaled cd.
        """
        self._alpha = float(alpha)
        self._gcd = -1.0
        k = self.k.numpy().astype(np.float64)
        from .mesh_contact import HC
        self.cd.assign((1.5 * self._alpha * k / HC).astype(np.float32))
        return self

    def calibrate_stiffness(self, pen_frac=0.005):
        """k_i so each body sinks pen_frac of its own size under its own weight.

        Forces are area-weighted and sum to 1 per body, so unlike the sphere cover there is
        no contact-count factor to measure -- only the fraction of the surface actually
        touching, which the weights already account for.
        """
        m = self.mass.numpy(); sz = np.asarray(self.sizes)
        self.k.assign((m * 9.81 / (pen_frac * sz)).astype(np.float32))
        if getattr(self, "_alpha", None) is not None:      # keep cd tied to the new k
            self.set_restitution_alpha(self._alpha)

    def rollout(self):
        self.pos[0].assign(self.pos0)
        self.rot[0].assign(self.rot0)
        self.vlin[0].assign(self.vlin0)
        self.vang[0].assign(self.vang0)
        gseq = self.gravity_seq
        for t in range(self.n_steps):
            g_t = self.gravity if gseq is None else wp.vec3(*gseq[t])
            self.force.zero_(); self.torque.zero_()
            wp.launch(world_verts, self.nV,
                      inputs=[self.pos[t], self.rot[t], self.local, self.body],
                      outputs=[self.world])
            wp.launch(ground_contact_mesh, self.nV,
                      inputs=[self.world, self.body, self.pos[t], self.vlin[t], self.vang[t],
                              self.ground_z, self.k, self.cd, self.mu, self._gmu, self._gcd,
                              self.weight, self.half_w, self.half_d,
                              wp.vec3(*[float(v) for v in self._gn]),
                              wp.vec3(*[float(v) for v in self._gt]),
                              wp.vec3(*[float(v) for v in self._gp])],
                      outputs=[self.force, self.torque])
            if self.N > 1:
                wp.launch(pair_contact_mesh, self.nV * self.N,
                          inputs=[self.world, self.body, self.mesh_ids, self.N,
                                  self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                                  self.k, self.cd, self.mu, self.weight, self.query_dist],
                          outputs=[self.force, self.torque])
            if self._link_arrays is not None:
                la, lb, pa, pb, ks, cs = self._link_arrays
                wp.launch(apply_links, len(self._links),
                          inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                                  la, lb, pa, pb, ks, cs],
                          outputs=[self.force, self.torque])
            wp.launch(integrate_6dof, self.N,
                      inputs=[self.pos[t], self.rot[t], self.vlin[t], self.vang[t],
                              self.force, self.torque, self.mass, self.inv_mass,
                              self.G, self.Ginv, g_t, self.dt],
                      outputs=[self.pos[t + 1], self.rot[t + 1],
                               self.vlin[t + 1], self.vang[t + 1]])
            wp.launch(apply_revolute, self.N,
                      inputs=[self.pos[t + 1], self.rot[t + 1], self.vlin[t + 1],
                              self.vang[t + 1], self.is_hinge, self.anchor, self.axis,
                              self._hinge_damp, self._hinge_limit, self._pivot_local])

    def rest_height(self, bi):
        """Table-relative z that puts this body's lowest vertex exactly on the table."""
        P = self._loc_np[self._bod_np == bi]
        return self.ground_z - float(P[:, 2].min())

    def positions(self, stride=60):
        return np.stack([self.pos[t].numpy() for t in range(0, self.n_steps + 1, stride)])

    def rotations(self, stride=60):
        return np.stack([self.rot[t].numpy() for t in range(0, self.n_steps + 1, stride)])
