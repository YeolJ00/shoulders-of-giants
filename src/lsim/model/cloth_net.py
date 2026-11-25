"""The learned simulator's cloth path (phase 3): the network predicts forces, an exact layer moves every cloth
vertex by its own mass.

Tokens (vertex, frame) as in lsim.model.net: the cloth's vertices, the gripper's points, static surface points
(table, floor, plank, the objects on the table). Edges, re-found every frame from that frame's positions:

  mesh     the cloth's mesh edges, with the rest offset and the strain
  contact  cloth vertex to a gripper or static point within 3 cm
  self     two cloth vertices within 2 cm whose rest positions are more than 3 cm apart (layers of a fold)

The same blocks as the rigid model (lsim.model.net.Block: time, space, object), the cloth's four slots on its body
token. Heads, forces in units of the vertex's weight m g:

  internal  one force per cloth vertex, made zero-sum over the cloth (its own forces -- elastic, bending, damping,
            self-contact -- cannot move its centre of mass); held vertices take their share, which the gripper
            absorbs
  contact   one force per cloth vertex that has a contact edge; none in free flight
  slots     clean values of the four slots (lsim.train.cloth_windows.CLOTH_PARAMS)

cloth_step (no weights): gravity; the teacher's air drag on each triangle's normal velocity (rho Cd = 1.44 kg/m^3,
exact); the forces over the vertex mass (w x area share, the density given or predicted); a step on frame-average
velocities (x' = x + v dt + a dt^2); then Coulomb contact with the static surfaces by default -- a vertex whose
step would end less than its radius above a surface it was above ends at its radius (inelastic: fabric does not
bounce), and friction removes up to mu times the surface's normal impulse from its sliding (mu the mean of the
cloth's slot and the surface's, the teacher's rule); held vertices follow the gripper (positions given). The
default holds a towel on the table and slides it by Coulomb's law with the network's outputs at zero, as v1's
did for rigid bodies.

Layer 2 (2026-10-20; ClothSimulator(layer=2)): the surface's normal impulse is all of it -- the default's velocity
change plus the network's contact force pushing the vertex out of the surface -- and the exact planes push a vertex
up to 1 cm inside them out to its radius. Layer 1 capped friction by the default's share alone and never pushed out:
the chunk noise sinks table vertices a few mm, where the default did not hold them, so the network learned to hold
every table vertex up itself (92-99% of its weight); the default was left almost no friction, and resting towels
slid 5-37 mm in a second under the model. A strain limit and cloth-on-cloth contact were tried as further layers and
removed (DIRECTION.md "v1 overnight"): the layer stays gravity, F = ma, the teacher's air drag, the grasp and contact
with fixed surfaces; how a cloth holds itself together is the network's to learn, not the layer's.

Shape restoration (2026-11-03, the user's decision; off by default, --restore / LSIM_RESTORE; restore_shape): every
learned model -- deterministic and generative, cloth, rope and soft -- failed one invariant over long rollouts (towels
shrank to ~10% of their area in 20 s, ropes stretched 1.4-14x in 4 s, soft bodies inverted ~50% of their tets) at
per-step errors of 1-4 mm, after every learned remedy (edge losses, long rollouts, a generative head). So one exact
rule: after the contact, each edge's length and each soft tet's volume are restored into the teachers' own bands
(RESTORE_BANDS), with what static friction holds fixed and the exact planes kept out of; the next frame-average
velocity follows from the restored positions. Everything else stays learned.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

from lsim.model.net import Block, mlp, rbf
from lsim.train.cloth_windows import C_CENTRE, C_SCALE, KIND, KIND_CENTRE, KIND_SCALE

G = 9.81
DT = 1.0 / float(os.environ.get("LSIM_FPS", 30.0))   # one frame (LSIM_FPS: the frame-rate study's 60, 120)
R_CONTACT = 0.03
R_SELF, R_SELF_REST = 0.02, 0.03
PAIR_CHUNK = 50_000_000                                            # candidate pairs per pass of pairs_within
RHO_CD = 1.44
TOUCH = 0.002                                      # m: contact counts this far above the vertex radius
PUSH_OUT = 0.01                                    # m: the exact planes push out a vertex up to this far inside
GRAVITY = torch.tensor([0.0, 0.0, -G])
# Shape restoration (2026-11-03, the user; off by default: model.restore None). Per material (cloth 0, rope 1, soft 2;
# lsim.train.cloth_windows.MATERIAL): each edge's strain L / L0 - 1 is pulled back inside [-compress, +stretch] and a
# soft body's tets' volume ratio V / V0 inside [1 - compress, 1 + expand]. The bands are the teachers' own
# (scripts/l2_strain_bands.py -> docs/results/l2_strain_bands.json; 320 cloth, 256 rope, 192 soft episodes, every
# third frame). A model's edges sit at the band's edge wherever its forces push them out, so the band is what its
# body strains: cloth 1% either way (the teacher's stretch p99.9 0.69%, compression 0.56%; 0.07% of its edge-frames
# beyond 1%), rope +3% / -1% (p99.99 2.85% / 1.05%: bungee cords stretch), soft tet edges 20% either way (0.25% of the
# teacher's edge-frames beyond, gel 0.75%, sponge 0.65%; p99.9 27%) and soft tet volumes -20% / +5% (0.62% / 0.05% of
# its tet-frames beyond; it inverts 4e-8). The soft bands are tighter than the teacher's tail on purpose: at 40% / -35%
# the mixed model's soft bodies still sank to 0.54 of their volume with 3% of tets inverted (8 held-out episodes; at
# 20% / -20% with 80 sweeps: 0.84 and 0.000) -- a squeezed gel loses a little of its last 0.5%.
RESTORE_BANDS = {"edge": {0: (0.01, 0.01), 1: (0.03, 0.01), 2: (0.2, 0.2)}, "vol": {2: (0.05, 0.2)}, "iters": 10,
                 "omega": 1.0, "pre": False, "jacobi": (2,), "iters_jacobi": 80, "omega_jacobi": 1.5,
                 "tet_cap": 0.25}


def pairs_within(x, key, src_mask, dst_mask, r):
    """Token pairs (i, j), i in src_mask, j in dst_mask, the same key (sample and frame), |x_i - x_j| < r, i != j,
    by spatial hashing (lsim.model.net.contact_edges, one direction)."""
    dev = x.device
    n = x.shape[0]
    tok = torch.arange(n, device=dev)
    cell = torch.floor(x / r).long()
    cell = cell - cell.min(0).values + 1
    span = cell.max(0).values + 2

    def key_of(kk, c):
        return ((kk * span[0] + c[:, 0]) * span[1] + c[:, 1]) * span[2] + c[:, 2]
    dk = key_of(key, cell)
    dst = tok[dst_mask]
    order = dst[torch.argsort(dk[dst_mask])]
    skey = dk[order]
    src = tok[src_mask]
    off = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=dev)
    s27 = src.repeat(len(off))                                          # all 27 neighbour cells in one pass (one
    k = key_of(key[s27], (cell[src][None] + off[:, None]).reshape(-1, 3))   # host sync, not 27: rollouts call this
    lo = torch.searchsorted(skey, k, side="left")                       # 4 times a frame)
    cnt = torch.searchsorted(skey, k, side="right") - lo
    cum = torch.cumsum(cnt, 0)
    if int(cum[-1]) <= PAIR_CHUNK:                                      # the usual case: one pass
        bounds = [0, len(cnt)]
    else:                                                               # a crumpled cloth or rope packs a cell with
        cuts = torch.searchsorted(cum, torch.arange(PAIR_CHUNK, int(cum[-1]), PAIR_CHUNK, device=dev))  # hundreds of
        bounds = [0] + sorted(set(cuts.clamp(1, len(cnt) - 1).tolist())) + [len(cnt)]  # vertices: candidates in chunks
    out = []                                                            # (one went to 10 GB and OOMed, 2026-11-03)
    for a, b in zip(bounds[:-1], bounds[1:]):
        c = cnt[a:b]
        tot = int(c.sum())
        i = torch.repeat_interleave(s27[a:b], c, output_size=tot)
        j = order[torch.repeat_interleave(lo[a:b] - torch.cumsum(c, 0) + c, c, output_size=tot) + torch.arange(tot, device=dev)]
        keep = (i != j) & ((x[i] - x[j]).norm(dim=-1) < r)
        out.append(torch.stack([i[keep], j[keep]]))
    return torch.cat(out, 1)


def cloth_edges(batch, pos):
    """Contact and self-contact edges of every frame (token = vertex * Fw + frame), both directions."""
    N, Fw, _ = pos.shape
    dev = pos.device
    x = pos.reshape(-1, 3)
    v = torch.arange(N * Fw, device=dev) // Fw
    key = batch["sample_of"][v] * Fw + torch.arange(N * Fw, device=dev) % Fw
    cl = batch["kind"][v] == KIND["cloth"]
    if "surf_v" in batch:                                          # a soft body's inside touches nothing
        cl = cl & batch["surf_v"][v]
    gone = batch["absent"].reshape(-1)
    ec = pairs_within(x, key, cl & ~gone, ~cl & ~gone, R_CONTACT)
    es = pairs_within(x, key, cl & ~gone, cl & ~gone, R_SELF)
    far = (batch["rest"][v[es[0]]] - batch["rest"][v[es[1]]]).norm(dim=-1) * batch["size"][batch["sample_of"][v[es[0]]]] > R_SELF_REST
    es = es[:, far]
    return torch.cat([ec, ec.flip(0)], 1), es                      # self-contact pairs come both ways already


class ClothAttention(nn.Module):
    """Every cloth vertex attends to every vertex of its cloth in the same frame. A towel is close to inextensible:
    its tension at a vertex depends on all of it within a frame (elastic waves cross it in about a millisecond),
    and six rounds of message passing reach 6 cm. Overfitting 16 episodes without it, moving cloth was 57 mm off
    after 10 frames, worse than constant velocity at short horizons."""

    def __init__(self, d, heads=4):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, h, idx, valid):
        G, L = idx.shape
        x = self.ln(h)[idx]                                          # (G, L, d)
        q, k, v = self.qkv(x).view(G, L, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        o = Fn.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        o = self.out(o.transpose(1, 2).reshape(G, L, -1))
        return h.index_add(0, idx[valid], o[valid].to(h.dtype))


def cloth_groups(batch, Fw):
    """Token ids of every (cloth, frame), padded: (n_bodies * Fw, longest cloth) and which are real."""
    dev = batch["pos"].device
    cl = torch.nonzero(batch["is_cloth"]).squeeze(1)                # cloth vertices, grouped by sample in order
    b = batch["body_of"][cl]
    nb = int(batch["n_bodies"])
    cnt = torch.bincount(b, minlength=nb)
    L = int(cnt.max())
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(len(cl), device=dev) - start[b]
    idx = torch.zeros(nb, Fw, L, dtype=torch.long, device=dev)
    valid = torch.zeros(nb, Fw, L, dtype=torch.bool, device=dev)
    f = torch.arange(Fw, device=dev)
    idx[b[:, None], f[None], rank[:, None]] = cl[:, None] * Fw + f[None]
    valid[b[:, None], f[None], rank[:, None]] = True
    return idx.view(nb * Fw, L), valid.view(nb * Fw, L)


class ClothSimulator(nn.Module):
    def __init__(self, d=128, layers=6, n_params=4, ckpt=True, bf16=True, causal=True, global_attn=True, prob_slots=False,
                 direct=False, unc=False, accel=False, layer=2):
        """layer: the exact layer's contact default it is trained with (2: friction capped by the whole normal
        impulse, planes push out; 1: the first runs, cloth_v1 and before) -- see the module docstring.
        prob_slots: the slot head gives a Gaussian per slot (mean and log variance, normalised units) -- the
        posterior over a hidden material given what was seen -- and a rollout samples from it.
        direct: baseline B2 for cloth -- the same network gives each vertex's whole acceleration (units of g; no
        gravity, drag, zero-sum or contact default: x' = 2x - x_prev + a dt^2, held vertices given), so the model
        against it isolates the exact layer.
        accel: accelerations as inputs (units of g): each vertex's over the last frame, and for the gripper's points and
        held cloth the one over the coming step (from the given action). Otherwise the network has to difference
        velocities of ~1 m/s across frames to find accelerations of ~0.1 g, in bf16 blocks.
        unc: a head for each free vertex's one-step uncertainty (the expected log of its position error, metres),
        on the network's state detached so it never moves the forces; trained by regression in log space on the
        model's own one-step error, sampled in rollouts as an isotropic Gaussian of that size so an ensemble spreads
        by what the model does not know (the teacher alone spreads 0.6 mm in a second)."""
        super().__init__()
        self.d, self.ckpt, self.bf16, self.n_params, self.prob_slots, self.direct = d, ckpt, bf16, n_params, prob_slots, direct
        self.layer = layer
        self.unc = mlp(d, d, 1) if unc else None
        self.gattn = nn.ModuleList([ClothAttention(d) for _ in range(layers)]) if global_attn else None
        # vel, rest, role, kind, held, given force (asinh), force state, given frame, present, tau, surface
        # material, static normal, log area share, the action over the coming step (velocity of the gripper's
        # points and the held cloth from this frame to the next: given; causal attention hid it, and the free cloth
        # moves with what the held cloth does within the frame)
        self.accel = accel
        self.vert_in = mlp(3 + 3 + 3 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 2 + 3 + 1 + 3 + (6 if accel else 0), d, d)
        # offset, dist, rbf, rel vel, kind (mesh, contact, self), rest offset, strain at two scales (a stiff towel
        # hangs at strains of 1e-5, a soft one stretches 1e-2), normal . offset
        self.edge_in = mlp(3 + 1 + 8 + 3 + 3 + 3 + 2 + 1, 32, 16)
        self.body_in = mlp(2 * n_params, d, d)
        self.blocks = nn.ModuleList([Block(d, causal=causal) for _ in range(layers)])
        self.f_int = mlp(d, d, 3)
        self.f_con = mlp(d, d, 3)
        self.param = mlp(2 * d, d, 2 * n_params if prob_slots else n_params)
        for head in (self.f_int, self.f_con):                     # start from the default: free fall + Coulomb
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, batch):
        """Per-vertex internal and contact forces in units of m g (N, Fw, 3) each, the slot predictions
        (n_bodies, 4), and what the physics layer needs about contacts."""
        pos, vel = batch["pos"], batch["vel"]
        N, Fw, _ = pos.shape
        dev = pos.device
        present = (~batch["absent"]).float()
        state = batch["f_state"]
        given = batch["frame_given"].float()
        f_given = torch.asinh(batch["f_ext"] * (state == 1)[..., None] / 0.01)
        log_area = torch.log(batch["area"].clamp_min(1e-9) / 1e-4) * batch["is_cloth"]
        feats = torch.cat([vel * present[..., None], batch["rest"][:, None].expand(-1, Fw, -1),
                           Fn.one_hot(batch["role"], 3).float()[:, None].expand(-1, Fw, -1),
                           Fn.one_hot(batch["kind"], 3).float()[:, None].expand(-1, Fw, -1),
                           batch["held"].float()[..., None], f_given, Fn.one_hot(state.long(), 3).float(),
                           given[None, :, None].expand(N, -1, -1), present[..., None],
                           batch["tau"][None, :, None].expand(N, -1, -1),
                           batch["surface"][:, None].expand(-1, Fw, -1), batch["normal"][:, None].expand(-1, Fw, -1),
                           log_area[:, None, None].expand(-1, Fw, -1), batch["kin_next"]], -1)
        if self.accel:                                               # (v_j - v_{j-1}) / dt, (v_{j+1} - v_j) / dt in g
            pv = torch.cat([vel[:, :1], vel[:, :-1]], 1)
            both = present * torch.cat([present[:, :1], present[:, :-1]], 1)
            a_prev = (vel - pv) / (DT * G) * both[..., None]
            kin = ((batch["kind"] == KIND["gripper"])[:, None] | batch["held"])[..., None]
            a_next = torch.where(kin, (batch["kin_next"] - vel) / (DT * G), torch.zeros_like(vel)) * present[..., None]
            feats = torch.cat([feats, a_prev, a_next], -1)
        h = self.vert_in(feats).reshape(N * Fw, -1)
        x, vt = pos.reshape(-1, 3), vel.reshape(-1, 3)
        em = batch["edges"].T
        em = torch.cat([em, em.flip(0)], 1)
        rest_off = (batch["rest"][em[0]] - batch["rest"][em[1]]) * batch["size"][batch["sample_of"][em[0]]][:, None]
        em = (em[:, :, None] * Fw + torch.arange(Fw, device=dev)).reshape(2, -1)
        rest_off = rest_off.repeat_interleave(Fw, 0)
        ec, es = cloth_edges(batch, pos)
        gone = batch["absent"].reshape(-1)
        keep = ~(gone[em[0]] | gone[em[1]])
        em, rest_off = em[:, keep], rest_off[keep]
        vtx = torch.arange(N * Fw, device=dev) // Fw
        e_all, f_all = [], []
        for e, kind in ((em, 0), (ec, 1), (es, 2)):
            off = x[e[0]] - x[e[1]]
            dist = torch.linalg.norm(off, dim=-1)
            ro = rest_off if kind == 0 else torch.zeros_like(off)
            rl = ro.norm(dim=-1)
            sr = torch.where(rl > 0, dist / rl.clamp_min(1e-9) - 1.0, torch.zeros_like(dist))
            strain = torch.stack([torch.asinh(1e2 * sr), torch.asinh(1e5 * sr)], -1)
            nd = (batch["normal"][vtx[e[1]]] * off).sum(-1) / R_CONTACT if kind == 1 else torch.zeros_like(dist)
            f_all.append(torch.cat([off / R_CONTACT, dist[:, None] / R_CONTACT, rbf(dist, r=R_CONTACT), vt[e[0]] - vt[e[1]],
                                    Fn.one_hot(torch.full_like(dist, kind).long(), 3).float(), ro / 0.01, strain,
                                    nd[:, None]], -1))
            e_all.append(e)
        e_idx = torch.cat(e_all, 1)
        e_feat = self.edge_in(torch.cat(f_all, 0))
        nb = int(batch["n_bodies"])
        body_of = batch["body_of"]
        tok_body = (body_of[:, None] * Fw + torch.arange(Fw, device=dev)[None]).reshape(-1)
        body_tok = self.body_in(torch.cat([batch["params"] * batch["param_given"], batch["param_given"].float()], -1))
        body_tok = body_tok[:, None].expand(-1, Fw, -1).reshape(nb * Fw, -1)
        absent = batch["absent"]
        gi, gv = cloth_groups(batch, Fw) if self.gattn is not None else (None, None)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            for k, blk in enumerate(self.blocks):
                if self.training and self.ckpt:
                    h = checkpoint(blk, h, Fw, e_idx, e_feat, tok_body, nb * Fw, body_tok, absent, use_reentrant=False)
                    if self.gattn is not None:
                        h = checkpoint(self.gattn[k], h, gi, gv, use_reentrant=False)
                else:
                    h = blk(h, Fw, e_idx, e_feat, tok_body, nb * Fw, body_tok, absent)
                    if self.gattn is not None:
                        h = self.gattn[k](h, gi, gv)
        h = h.float()
        cl_tok = batch["is_cloth"].repeat_interleave(Fw)
        w_tok = (present * given[None, :]).reshape(-1) * cl_tok
        pooled = torch.zeros(nb, self.d, device=dev).index_add_(0, body_of.repeat_interleave(Fw), h * w_tok[:, None])
        cnt = torch.zeros(nb, device=dev).index_add_(0, body_of.repeat_interleave(Fw), w_tok)
        params_pred = self.param(torch.cat([pooled / cnt.clamp_min(1)[:, None], body_tok.view(nb, Fw, -1)[:, 0]], -1))
        if self.prob_slots:                                          # mean, log variance (clamped: 0.007-7 units)
            self.slot_logvar = params_pred[:, self.n_params:].clamp(-10.0, 4.0)
            params_pred = params_pred[:, :self.n_params]
        if self.unc is not None:                                     # log rms one-step error per (vertex, frame)
            self.log_sigma = self.unc(h.detach()).view(N, Fw) - 7.0     # starts near 1 mm
        f_int = self.f_int(h).view(N, Fw, 3) * batch["is_cloth"][:, None, None]
        if self.direct:                                              # B2: the whole acceleration, nothing exact
            return f_int, torch.zeros_like(f_int), params_pred, ec
        has_c = torch.zeros(N * Fw, device=dev).index_add_(0, ec[0], torch.ones(ec.shape[1], device=dev)) > 0
        f_con = self.f_con(h).view(N, Fw, 3) * (has_c.view(N, Fw) & batch["is_cloth"][:, None])[..., None]
        # internal forces sum to zero over a cloth no one holds (mass-weighted: no net force on it); a held cloth's
        # free part is pulled by the held vertices, which the gripper keeps where they are
        key = (body_of[:, None] * Fw + torch.arange(Fw, device=dev)[None]).reshape(-1)
        a = (batch["area"] * batch["is_cloth"]).repeat_interleave(Fw)
        tot = torch.zeros(nb * Fw, 3, device=dev).index_add_(0, key, f_int.reshape(-1, 3) * a[:, None])
        atot = torch.zeros(nb * Fw, device=dev).index_add_(0, key, a).clamp_min(1e-12)
        held_any = torch.zeros(nb * Fw, device=dev).index_add_(0, key, batch["held"].reshape(-1).float()) > 0
        mean = torch.where(held_any[:, None], torch.zeros_like(tot), tot / atot[:, None])
        f_int = (f_int.reshape(-1, 3) - mean[key]).view(N, Fw, 3) * batch["is_cloth"][:, None, None]
        return f_int, f_con, params_pred, ec


def kin_next(batch, pos_next):
    """The given action over each coming step, per (vertex, frame): (x_{j+1} - x_j) / dt for the gripper's points and
    the cloth held in step j, zero elsewhere. pos_next: (N, Fw + 1, 3) positions of the window's frames and the one
    after (only kinematic tokens' entries are read)."""
    kin = (batch["kind"] == KIND["gripper"])[:, None] | batch["held"]
    v = (pos_next[:, 1:] - pos_next[:, :-1]) / DT
    return torch.where(kin[..., None], v, torch.zeros_like(v))


def static_contact(batch, x_now, x_next):
    """For each cloth vertex: the static point nearest to its step's end or start (within 3 cm of either) -- the
    surface the step may run into -- or -1. x_now, x_next: (N, 3) at one frame."""
    dev = x_now.device
    N = x_now.shape[0]
    cl = batch["is_cloth"]
    st = (batch["kind"] == KIND["static"]) & batch["point_contact"]
    best_d = torch.full((N,), 1e9, device=dev)
    best_j = torch.full((N,), -1, device=dev, dtype=torch.long)
    for xa in (x_next, x_now):
        xx = torch.where(cl[:, None], xa, x_now)                     # static points stay where they are
        e = pairs_within(xx, batch["sample_of"], cl, st, R_CONTACT)
        if e.shape[1] == 0:
            continue
        d = (xx[e[0]] - x_now[e[1]]).norm(dim=-1)
        o = torch.argsort(d)                                          # nearest per vertex: by distance, then
        i, j, dd = e[0][o], e[1][o], d[o]                             # stably by vertex, first of each
        s = torch.argsort(i, stable=True)
        i, j, dd = i[s], j[s], dd[s]
        first = torch.cat([torch.ones(1, dtype=torch.bool, device=dev), i[1:] != i[:-1]])
        i, j, dd = i[first], j[first], dd[first]
        better = dd < best_d[i]
        best_d[i[better]], best_j[i[better]] = dd[better], j[better]
    return best_j


def _band(table, mat, default):
    """Per element of material ids mat: its entry of a {material: value} table (default where missing)."""
    out = torch.full(mat.shape, float(default), dtype=torch.float64, device=mat.device)
    for k, v in table.items():
        out = torch.where(mat == int(k), torch.as_tensor(float(v), dtype=torch.float64, device=mat.device), out)
    return out


def _groups(col):
    """Index sets of each colour (ascending)."""
    if col.numel() == 0:
        return []
    o = torch.argsort(col, stable=True)
    cnt = torch.bincount(col).tolist()
    return [g for g in torch.split(o, cnt) if len(g)]


def restore_shape(batch, x, fixed, m, cfg=None, clamp=None):
    """The exact layer's shape restoration (2026-11-03): positions x (N, 3) moved so that each mesh edge's length is back
    inside its band [L0 (1 - compress), L0 (1 + stretch)] (L0 the rest length; cloth mesh edges, rope segments, a soft
    body's tet edges) and each of a soft body's tets' volume inside [V0 (1 - compress), V0 (1 + expand)]
    (RESTORE_BANDS). Position-based (PBD): a constraint beyond its band is corrected along its gradient, shared by
    inverse mass (fixed vertices -- held, given, not deformable, stuck by static friction -- do not move), so every
    correction is an internal force pair (no net impulse). Cloth and rope edges: Gauss-Seidel over colours (batch
    edge_color: no two edges of a colour share a vertex), cfg["iters"] sweeps; soft bodies (cfg["jacobi"]) and all tet
    volumes: averaged Jacobi, cfg["iters_jacobi"] sweeps over-relaxed by cfg["omega_jacobi"], each constraint's
    correction over the most active constraints at any of its vertices (a tet mesh takes ~20 edge and ~40 tet colours:
    colour by colour cost 186 ms a step, 80 sweeps of this ~120). Differentiable (torch ops throughout). A constraint inside its
    band is untouched, so a body the teacher would leave alone is left alone. clamp: a function applied to the
    positions after every sweep (cloth_step: plane_keeper, so the sweeps slide a vertex along a surface instead of
    through it)."""
    cfg = RESTORE_BANDS if cfg is None else cfg
    E = batch["edges"]
    if E.numel() == 0:
        return x
    if cfg.get("st"):                                                    # straight-through: the corrections in the
        cfg = dict(cfg, st=False)                                        # forward pass, the identity in the backward
        with torch.no_grad():
            y = restore_shape(batch, x.detach(), fixed, m, cfg, clamp)
        return x + (y - x.detach())
    dt_ = x.dtype
    i, j = E[:, 0], E[:, 1]
    L0 = ((batch["rest"][i] - batch["rest"][j]).norm(dim=-1) * batch["size"][batch["sample_of"][i]]).to(dt_)
    mat_v = batch["material"][batch["body_of"]] if "material" in batch else torch.zeros_like(batch["body_of"])
    st = _band({k: v[0] for k, v in cfg["edge"].items()}, mat_v[i], 1e9).to(dt_)
    co = _band({k: v[1] for k, v in cfg["edge"].items()}, mat_v[i], 1.0).to(dt_)
    lo, hi = L0 * (1.0 - co), L0 * (1.0 + st)
    w = torch.where(fixed, torch.zeros_like(m), 1.0 / m.clamp_min(1e-12)).to(dt_)
    if "edge_color" not in batch or batch["edge_color"].shape[0] != E.shape[0]:
        from lsim.train.cloth_windows import greedy_colors
        batch["edge_color"] = torch.as_tensor(greedy_colors(E.cpu().numpy()), device=E.device)
    T = batch.get("tets")
    tg = []
    if T is not None and T.numel() and cfg.get("vol"):
        def vol(p):
            a = p[T[:, 0]]
            return torch.linalg.det(torch.stack([p[T[:, 1]] - a, p[T[:, 2]] - a, p[T[:, 3]] - a], -1)) / 6.0
        rest_t = batch["rest"] * batch["size"][batch["sample_of"]][:, None]
        V0 = vol(rest_t.to(dt_))
        sgn = torch.sign(V0)
        ve = _band({k: v[0] for k, v in cfg["vol"].items()}, mat_v[T[:, 0]], 1e9).to(dt_)
        vc = _band({k: v[1] for k, v in cfg["vol"].items()}, mat_v[T[:, 0]], 1e9).to(dt_)
        vlo, vhi = V0.abs() * (1.0 - vc), V0.abs() * (1.0 + ve)
        tg = [True]                                                      # (solved by Jacobi: no tet colours needed)
        wt = w[T]                                                        # (T, 4)
    om = float(cfg.get("omega", 1.0))
    jac = cfg.get("jacobi", (2,))                                        # materials solved by averaged Jacobi
    je = torch.zeros_like(i, dtype=torch.bool)
    for mm in jac:
        je = je | (mat_v[i] == int(mm))
    # Gauss-Seidel over colours (cloth, rope: few colours) -- per colour, what does not change over the sweeps, once
    egs = []
    gs_e = torch.nonzero(~je).squeeze(1)
    for g in _groups(batch["edge_color"][gs_e]) if len(gs_e) else []:
        g = gs_e[g]
        ii, jj = i[g], j[g]
        wi, wj = w[ii], w[jj]
        k = torch.where(wi + wj > 0, om / (wi + wj).clamp_min(1e-30), torch.zeros_like(wi))
        egs.append((ii, jj, torch.cat([ii, jj]), torch.cat([-k * wi, k * wj])[:, None], lo[g], hi[g]))
    # averaged Jacobi (soft bodies: a tet mesh takes ~20 edge and ~40 tet colours, 186 ms a step colour by colour):
    # every constraint at once, each vertex's corrections averaged over its active constraints, over-relaxed
    jt = torch.nonzero(je).squeeze(1)
    jac_on = len(jt) > 0 or len(tg) > 0
    if jac_on:
        omj = float(cfg.get("omega_jacobi", 1.5))
        ji, jj_ = i[jt], j[jt]
        jwi, jwj = w[ji], w[jj_]
        jk = torch.where(jwi + jwj > 0, 1.0 / (jwi + jwj).clamp_min(1e-30), torch.zeros_like(jwi))
        jij = torch.cat([ji, jj_])
        jcoef = torch.cat([-jk * jwi, jk * jwj])[:, None]
        jlo, jhi = lo[jt], hi[jt]
        if len(tg):
            Tf = T.reshape(-1)
            tact = (wt > 0).any(-1)
            # a tet's correction moves no corner by more than tet_cap x its rest size (V0^(1/3)): near a flat or deeply
            # inverted tet the volume's gradient vanishes and -C / |grad|^2 is unbounded (zero output, a soft body's
            # inside falling through itself: 100 m in 30 frames without the cap)
            tcap = float(cfg.get("tet_cap", 0.25)) * V0.abs().clamp_min(1e-18) ** (1.0 / 3.0)
            pr_ = rest_t.to(dt_)[T]                                      # |grad V|^2 at rest: the floor of the
            r1, r2, r3 = pr_[:, 1] - pr_[:, 0], pr_[:, 2] - pr_[:, 0], pr_[:, 3] - pr_[:, 0]   # denominator (a flat tet's
            h1, h2, h3 = torch.cross(r2, r3, dim=-1), torch.cross(r3, r1, dim=-1), torch.cross(r1, r2, dim=-1)  # vanishes)
            gr0 = torch.stack([-(h1 + h2 + h3), h1, h2, h3], 1) / 6.0
            den0 = (wt * (gr0 ** 2).sum(-1)).sum(-1)
    n_it = int(cfg.get("iters", 10))
    for it in range(max(n_it, int(cfg.get("iters_jacobi", 80)) if jac_on else n_it)):
        if jac_on and it < int(cfg.get("iters_jacobi", 80)):
            # each constraint's correction over the largest number of active constraints at any of its vertices (not
            # each vertex's sum over its own count: that breaks the pairs' balance, so momentum, by 1e-4)
            cnt = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            if len(tg):                                                  # volumes: C = s V - clamp(s V, lo, hi)
                p = x[T]
                d1, d2, d3 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]
                g1 = torch.cross(d2, d3, dim=-1)
                g2 = torch.cross(d3, d1, dim=-1)
                g3 = torch.cross(d1, d2, dim=-1)
                gr = torch.stack([-(g1 + g2 + g3), g1, g2, g3], 1) * (sgn / 6.0)[:, None, None]
                sv = sgn * (d1 * g1).sum(-1) / 6.0
                C = sv - torch.clamp(sv, vlo, vhi)
                den = (wt * (gr ** 2).sum(-1)).sum(-1)
                ton = tact & (den > 0) & (C != 0)
                lam = torch.where(ton, -C / torch.maximum(den, 0.1 * den0), torch.zeros_like(C))
                cnt = cnt.index_add(0, Tf, (ton[:, None] & (wt > 0)).to(x.dtype).reshape(-1))
            if len(jt):                                                  # edges: C = L - clamp(L, lo, hi)
                d = x[ji] - x[jj_]
                L = d.norm(dim=-1)
                Lc = torch.clamp(L, jlo, jhi)
                c = ((L - Lc) / torch.maximum(L, 0.25 * jlo))[:, None] * d     # (a near-zero edge: bounded slope)
                eon = ((L != Lc) & (jk > 0)).to(x.dtype)
                cnt = cnt.index_add(0, jij, torch.cat([eon, eon]))
            cnt = cnt.clamp_min(1.0)
            dx = torch.zeros_like(x)
            if len(tg):
                sc = omj / cnt[T].max(-1).values
                dt4 = ((lam * sc)[:, None] * wt)[..., None] * gr               # (T, 4, 3)
                big = dt4.norm(dim=-1).max(-1).values
                dt4 = dt4 * (tcap / big.clamp_min(1e-30)).clamp(max=1.0)[:, None, None]
                dx = dx.index_add(0, Tf, dt4.reshape(-1, 3))
            if len(jt):
                sc = omj / torch.maximum(cnt[ji], cnt[jj_])
                cs = c * sc[:, None]
                dx = dx.index_add(0, jij, torch.cat([cs, cs]) * jcoef)
            x = x + dx
        if it < n_it:
            for ii, jj, ij, coef, el, eh in egs:                         # edges: C = L - clamp(L, lo, hi)
                d = x[ii] - x[jj]
                L = d.norm(dim=-1)
                # over max(L, lo / 4), not L: an edge crushed to near zero length (a crumpled cloth) made d / L's
                # derivative ~1 / L, and a 60-frame rollout's gradient 2.5e6 where it is 30 without the restoration
                c = ((L - torch.clamp(L, el, eh)) / torch.maximum(L, 0.25 * el))[:, None] * d
                x = x.index_add(0, ij, torch.cat([c, c]) * coef)
        if clamp is not None:
            x = clamp(x)
    return x


def restore_config(spec):
    """The restoration a checkpoint's / the trainer's --restore spec asks for: "" or "0" off (None); "1" the measured
    bands (RESTORE_BANDS); otherwise "key=value,..." over them (cloth=s:c, rope=s:c, soft=s:c edge bands, vol=e:c,
    iters=n, omega=w)."""
    if spec in (None, "", "0", 0, False):
        return None
    import copy
    cfg = copy.deepcopy(RESTORE_BANDS)
    if str(spec) not in ("1", "True", "true", "on"):
        names = {"cloth": 0, "rope": 1, "soft": 2}
        for kv in str(spec).split(","):
            k, v = kv.split("=")
            if k in names:
                s_, c_ = v.split(":")
                cfg["edge"][names[k]] = (float(s_), float(c_))
            elif k == "vol":                                         # vol=expand:compress
                e_, c_ = v.split(":")
                cfg["vol"] = {2: (float(e_), float(c_))}
            elif k == "iters":
                cfg["iters"] = int(v)
            elif k == "st":                                           # straight-through gradient everywhere
                cfg["st"] = bool(int(v))
            elif k == "st_roll":                                      # ... in the long-rollout loss only
                cfg["st_roll"] = bool(int(v))
            elif k == "tet_cap":
                cfg["tet_cap"] = float(v)
            elif k == "iters_jacobi":
                cfg["iters_jacobi"] = int(v)
            elif k == "omega_jacobi":
                cfg["omega_jacobi"] = float(v)
            elif k == "jacobi":                                       # materials by Jacobi, e.g. jacobi=2 or jacobi=
                cfg["jacobi"] = tuple(int(c) for c in v.split(":") if c)
            elif k == "omega":
                cfg["omega"] = float(v)
            elif k == "pre":                                          # the pass before the contact (1) or not (0)
                cfg["pre"] = bool(int(v))
            else:
                raise ValueError(f"--restore: unknown key {k}")
    return cfg


def cloth_step(batch, x_prev, x_now, f, w_cloth, given_next, x_given_next, mu_cloth, dt=DT, drag_passes=2, f_contact=None,
               push_out=PUSH_OUT, restore=None):
    """One frame for every cloth vertex from two positions (N, 3) (frame-average velocity (x_now - x_prev) / dt):
    gravity, air drag, the forces f (N, 3, in newtons), the Coulomb default against static surfaces. Vertices in
    given_next (held cloth, the gripper, static points) end at x_given_next. w_cloth, mu_cloth: per body
    (n_bodies,). f_contact: the part of f that is the network's contact force (N, 3, newtons); its push out of a
    surface the vertex touches counts toward that surface's normal impulse, which caps friction (None: the default's
    impulse alone, layer 1). push_out: how deep inside an exact plane a vertex is still pushed out to its radius (0:
    never, layer 1). restore: a restore_shape configuration (None: off, the default) -- after the contact, edge lengths
    and tet volumes restored into their bands with the vertices static friction holds (and held ones) fixed, so the
    restoration never drags a resting vertex (2026-10-21's strain limit, run after the contact with every vertex free,
    slid overhanging towels off the table; run before it, as then, the lip of a towel over the table's edge took the
    hanging part's whole pull and slid), and kept out of the exact planes (plane_keeper); restore["pre"] adds the
    before-the-contact pass too. Positions only: the next frame-average velocity follows from them. Returns the next
    positions (N, 3)."""
    dev = x_now.device
    cl = batch["is_cloth"]
    body = batch["body_of"]
    m = (w_cloth[body] * batch["area"]).clamp_min(1e-9)              # vertex mass, kg
    v = (x_now - x_prev) / dt
    # air drag on each triangle's normal velocity, at all three corners (Newton's semi-implicit kernel), implicit:
    # each vertex's drag -(rho Cd / 6) sum A |n . v_tri| v_tri is taken as a damping c v with c frozen at the step's
    # velocity (two passes), v' = (v + a0 dt + (drag + c v) dt / m) / (1 + c dt / m). A light towel whipping at 4 m/s
    # is slowed at ~23 g (time constant 18 ms, under a frame), where an explicit step overshoots: the layer given the
    # teacher's exact forces missed by up to 7 mm in the air
    T = batch["tris"]
    a_, b_, c_ = x_now[T[:, 0]], x_now[T[:, 1]], x_now[T[:, 2]]
    cr = torch.cross(b_ - a_, c_ - a_, dim=-1)
    area = 0.5 * cr.norm(dim=-1).clamp_min(1e-12)
    n = cr / (2.0 * area[:, None])
    a0 = GRAVITY.to(dev) + f / m[:, None]
    # a rope's air drag (lsim.teacher.rope_teacher._air_drag, exact): the cross-flow drag of a cylinder on each
    # segment's normal velocity, -1/2 rho Cd (2 r) L |v_n| v_n, half to each end, implicit the same way (a soft body's
    # teacher has no air drag and gives no drag triangles)
    S = batch.get("segs")
    rope_drag = S is not None and S.shape[0] > 0
    if rope_drag:
        ts = x_now[S[:, 1]] - x_now[S[:, 0]]
        Ls = ts.norm(dim=-1).clamp_min(1e-9)
        ts = ts / Ls[:, None]
        ks = 0.5 * RHO_CD * 2.0 * batch["seg_r"] * Ls                   # F = -ks |v_n| v_n per segment
    vd = v
    for _ in range(drag_passes):
        vm = (vd[T[:, 0]] + vd[T[:, 1]] + vd[T[:, 2]]) / 3.0
        ct = (RHO_CD / 6.0) * area * (n * vm).sum(-1).abs()            # per triangle, at each of its corners
        fd = -ct[:, None] * vm
        drag = torch.zeros_like(x_now)
        cv = x_now.new_zeros(x_now.shape[0])
        for k in range(3):
            drag = drag.index_add(0, T[:, k], fd)
            cv = cv.index_add(0, T[:, k], ct)
        if rope_drag:
            vs = 0.5 * (vd[S[:, 0]] + vd[S[:, 1]])
            vn = vs - (vs * ts).sum(-1, keepdim=True) * ts
            cs = 0.5 * ks * vn.norm(dim=-1)                              # half to each end
            for k in range(2):
                drag = drag.index_add(0, S[:, k], -cs[:, None] * vn)
                cv = cv.index_add(0, S[:, k], cs)
        h = cv * dt / m
        vd = (v + a0 * dt + (drag + cv[:, None] * vd) * dt / m[:, None]) / (1.0 + h)[:, None]
    v1 = vd
    x1 = x_now + v1 * dt
    if restore is not None and restore.get("pre", True):               # edge lengths / volumes back in their bands
        x1 = torch.where(given_next[:, None], x_given_next, x1)
        x1 = plane_keeper(batch, x_now, batch["radius"][batch["sample_of"]])(
            restore_shape(batch, x1, given_next | ~cl, m, restore))
        v1 = torch.where(cl[:, None], (x1 - x_now) / dt, v1)
    # Coulomb contact against static surfaces (inelastic, mu the mean of the cloth's and the surface's): the flat ones
    # (table top, floor, plank top) as exact rectangles, the objects as their points and normals; per vertex the
    # surface it touches that needs the larger change of its normal velocity
    r = batch["radius"][batch["sample_of"]]
    vn_p, need_p, nrm_p, mus_p, hit_p = _point_contact(batch, x_now, x1, v1, r, dt)
    vn_q, need_q, nrm_q, mus_q, hit_q = _plane_contact(batch, x_now, x1, v1, r, dt, push_out)
    use_q = hit_q & (~hit_p | (need_q.clamp_min(0.0) >= need_p.clamp_min(0.0)))
    hit = (hit_p | hit_q) & cl
    if "surf_v" in batch:                                               # a soft body's inside touches nothing
        hit = hit & batch["surf_v"]
    nrm = torch.where(use_q[:, None], nrm_q, nrm_p)
    vn = torch.where(use_q, vn_q, vn_p)
    need = torch.where(use_q, need_q, need_p) * hit                     # normal velocity change that ends at r
    dvn = need.clamp_min(0.0)                                           # the default's: it never pulls
    # the surface's normal impulse (as a velocity change) is what would stop the vertex at the surface had the
    # network's contact force not pushed it out: the default supplies dvn, the network the rest (no more than that:
    # a surface cannot hold a vertex the network lifts off it). Friction does not care which part holds it up
    held_up = dvn
    if f_contact is not None:
        push = ((f_contact * nrm).sum(-1) / m).clamp_min(0.0) * dt
        held_up = (need + push).clamp_min(0.0) * hit
    vt = v1 - vn[:, None] * nrm
    vts = vt.norm(dim=-1).clamp_min(1e-12)
    mu = 0.5 * (mu_cloth[body] + torch.where(use_q, mus_q, mus_p))
    keep_t = (1.0 - mu * held_up / vts).clamp(0.0, 1.0)
    v1 = torch.where((held_up > 0)[:, None], vt * keep_t[:, None] + (vn + dvn)[:, None] * nrm, v1)
    x1 = torch.where(cl[:, None], x_now + v1 * dt, x_now)
    x1 = torch.where(given_next[:, None], x_given_next, x1)
    if restore is not None:
        # friction may have taken back what the first pass moved on the surface (a towel over the table's edge: the
        # lip vertex held, its edge to the hanging part 440% long with one pass); a second pass with every vertex a
        # surface holds fixed restores the rest, and a vertex it would move into a flat surface it was above ends on it
        still = given_next | ~cl | (hit & (held_up > 0) & (keep_t <= 0))      # held, or stuck by static friction
        x1 = restore_shape(batch, x1, still, m, restore, clamp=plane_keeper(batch, x1, r))
    return x1


def tile_batch(batch, K, restore=False):
    """What cloth_step reads of a batch, repeated for K frames stepped at once: token k * N + i is vertex i in the
    k-th copy (its own hashing key, triangles shifted)."""
    N = batch["is_cloth"].shape[0]
    S = int(batch["n_bodies"])
    dev = batch["is_cloth"].device
    out = {k: batch[k].repeat(K, *([1] * (batch[k].dim() - 1))) for k in ("is_cloth", "body_of", "area", "kind", "normal",
                                                                            "surface", "spacing")}
    out["sample_of"] = (batch["sample_of"][None] + S * torch.arange(K, device=dev)[:, None]).reshape(-1)
    out["tris"] = (batch["tris"][None] + N * torch.arange(K, device=dev)[:, None, None]).reshape(-1, 3)
    out["radius"] = batch["radius"].repeat(K)
    out["point_contact"] = batch["point_contact"].repeat(K)
    if "surf_v" in batch:
        out["surf_v"] = batch["surf_v"].repeat(K)
    if "segs" in batch:
        out["segs"] = (batch["segs"][None] + N * torch.arange(K, device=dev)[:, None, None]).reshape(-1, 2)
        out["seg_r"] = batch["seg_r"].repeat(K)
    out["planes"], out["plane_sample"], out["n_samples"] = batch["planes"], batch["plane_sample"], S
    out["n_bodies"] = S
    if restore:                                                         # what restore_shape reads
        out["edges"] = (batch["edges"][None] + N * torch.arange(K, device=dev)[:, None, None]).reshape(-1, 2)
        out["rest"] = batch["rest"].repeat(K, 1)
        out["size"] = batch["size"].repeat(K)
        if "material" in batch:
            out["material"] = batch["material"]                         # indexed by body_of (not offset)
        if "edge_color" in batch:
            out["edge_color"] = batch["edge_color"].repeat(K)
        if "tets" in batch:
            out["tets"] = (batch["tets"][None] + N * torch.arange(K, device=dev)[:, None, None]).reshape(-1, 4)
            out["tet_color"] = batch["tet_color"].repeat(K)
    return out


def direct_step(x_prev, x_now, acc_g, given_next, x_given_next, dt=DT):
    """Baseline B2's step: x' = 2x - x_prev + a dt^2 (a in units of g), given vertices at their given positions."""
    x = 2.0 * x_now - x_prev + acc_g * (G * dt * dt)
    return torch.where(given_next[..., None], x_given_next, x)


def step_frames(batch, x_prev, x_now, f, w, given_next, x_given_next, mu, tiled=None, f_contact=None, push_out=PUSH_OUT,
                restore=None):
    """cloth_step for K frames at once: arrays (N, K, 3) / (N, K) -> next positions (N, K, 3)."""
    N, K, _ = x_now.shape
    tb = tile_batch(batch, K, restore=restore is not None) if tiled is None else tiled
    flat = lambda a: a.transpose(0, 1).reshape(K * N, *a.shape[2:])
    x = cloth_step(tb, flat(x_prev), flat(x_now), flat(f), w, flat(given_next), flat(x_given_next), mu,
                   f_contact=None if f_contact is None else flat(f_contact), push_out=push_out, restore=restore)
    return x.view(K, N, 3).transpose(0, 1)


def plane_keeper(batch, x_ref, r, depth=PUSH_OUT):
    """A function x -> x for the restoration's sweeps, keeping vertices out of the exact planes' slabs (table top, floor,
    plank top; down to `depth` below each, where the contact default would push them out): a vertex at or above its
    radius over the plane in x_ref ends at its radius (moved along the normal); a vertex beside the plane in x_ref (a
    towel's part hanging over the table's edge) that a sweep pulled under its edge goes back out beside it (the slab's
    side face: the exact layer has none, and the push-out would otherwise lift a hanging flap onto the table). What
    depends on x_ref alone is computed once."""
    PL, ps = batch["planes"], batch["plane_sample"]
    S = int(batch.get("n_samples", batch["n_bodies"]))
    dev = x_ref.device
    pid = torch.full((S, 3), -1, dtype=torch.long, device=dev)
    rank = torch.arange(len(ps), device=dev) - torch.searchsorted(ps, ps)
    pid[ps, rank] = torch.arange(len(ps), device=dev)
    cand = pid[batch["sample_of"] % S]
    cl = batch["is_cloth"]
    if "surf_v" in batch:
        cl = cl & batch["surf_v"]
    planes = []
    for k in range(3):
        q = cand[:, k]
        ok = (q >= 0) & cl
        if not bool(ok.any()):
            continue
        P = PL[q.clamp_min(0)]
        o, n, u, v, a, b = P[:, 0:3], P[:, 3:6], P[:, 6:9], P[:, 9:12], P[:, 12], P[:, 13]
        d0 = x_ref - o
        u0, v0, s0 = (d0 * u).sum(-1), (d0 * v).sum(-1), (d0 * n).sum(-1)
        in0 = (u0.abs() <= a) & (v0.abs() <= b)
        top = ok & in0 & (s0 >= r - 1e-6)
        side = ok & ~in0 & (s0 < r)
        out_u = u0.abs() > a
        planes.append((o, n, u, v, a, b, top, side, out_u, torch.sign(u0) * (a + r), torch.sign(v0) * (b + r)))

    def keep(x):
        for o, n, u, v, a, b, top, side, out_u, ut, vt in planes:
            d = x - o
            u1, v1, s1 = (d * u).sum(-1), (d * v).sum(-1), (d * n).sum(-1)
            slab = (u1.abs() <= a) & (v1.abs() <= b) & (s1 < r) & (s1 > -depth)
            up = (slab & top).to(x.dtype)[:, None] * (r - s1)[:, None] * n
            lat = torch.where(out_u[:, None], (ut - u1)[:, None] * u, (vt - v1)[:, None] * v)
            x = x + up + (slab & side).to(x.dtype)[:, None] * lat
        return x
    return keep


def _point_contact(batch, x_now, x1, v1, r, dt):
    """The objects' points: the nearest to the step's start or end. A step that ends within r + TOUCH of the point's
    tangent plane, from at or above the vertex's resting height there (within 1 mm of it) and over the point (within
    half its spacing), stops at r: a vertex lower than that is beside an edge or on a surface the points and normals
    place a few mm off (an object's learner points are not its collision mesh); there the default pushed resting
    vertices out by 6 mm given the teacher's exact forces, so it leaves them alone. Never pushes out. Returns the normal
    velocity, the change of it that would end the step at r (negative: the vertex leaves), the normal, the surface's
    friction and whether the vertex touches it, per vertex."""
    j = static_contact(batch, x_now, x1)
    has = j >= 0
    jj = j.clamp_min(0)
    nrm = batch["normal"][jj]
    p = x_now[jj]                                                    # static points do not move
    s0 = ((x_now - p) * nrm).sum(-1)
    s1 = ((x1 - p) * nrm).sum(-1)
    over = (x1 - p - s1[:, None] * nrm).norm(dim=-1) < 0.5 * batch["spacing"][jj]
    hit = has & over & (s1 < r + TOUCH) & (s0 > r - 0.001)
    vn = (v1 * nrm).sum(-1)
    need = torch.where(hit, torch.clamp(r - s0, max=0.0) / dt - vn, torch.zeros_like(vn))
    return vn, need, nrm, torch.exp(batch["surface"][jj, 0]), hit


def _plane_contact(batch, x_now, x1, v1, r, dt, push_out=PUSH_OUT):
    """The flat surfaces as exact rectangles (lsim.train.cloth_windows.ClothEpisode.planes): the same rule as
    _point_contact, over the rectangle itself (a 1 cm lattice of points held cloth up to 7 mm beyond the table's edge
    and made static points half a window's tokens), except that a vertex up to push_out inside the plane is pushed
    out to its radius within the step (the teacher's contact, 2e7 / s^2 critically damped, does it in a few of its
    substeps; the geometry is exact, unlike an object's points). Up to three planes per sample."""
    PL, ps = batch["planes"], batch["plane_sample"]
    S = int(batch.get("n_samples", batch["n_bodies"]))
    dev = x_now.device
    pid = torch.full((S, 3), -1, dtype=torch.long, device=dev)
    rank = torch.arange(len(ps), device=dev) - torch.searchsorted(ps, ps)       # planes are grouped by sample
    pid[ps, rank] = torch.arange(len(ps), device=dev)
    cand = pid[batch["sample_of"] % S]                                 # (N, 3)
    best_need = x_now.new_zeros(x_now.shape[0])
    best_vn, best_n = torch.zeros_like(best_need), torch.zeros_like(x_now)
    best_mu = torch.ones_like(best_need)
    best_hit = torch.zeros_like(best_need, dtype=torch.bool)
    for k in range(3):
        q = cand[:, k]
        valid = q >= 0
        P = PL[q.clamp_min(0)]
        o, n, u, v, a, b = P[:, 0:3], P[:, 3:6], P[:, 6:9], P[:, 9:12], P[:, 12], P[:, 13]
        s0 = ((x_now - o) * n).sum(-1)
        s1 = ((x1 - o) * n).sum(-1)
        d1 = x1 - o
        inside = ((d1 * u).sum(-1).abs() <= a) & ((d1 * v).sum(-1).abs() <= b)
        hit = valid & inside & (s1 < r + TOUCH) & (s0 > r - max(push_out, 0.001))
        vn = (v1 * n).sum(-1)
        gap = r - s0 if push_out > 0 else torch.clamp(r - s0, max=0.0)        # the normal travel that ends at r
        need = torch.where(hit, gap / dt - vn, torch.zeros_like(vn))
        better = hit & (~best_hit | (need.clamp_min(0.0) > best_need.clamp_min(0.0)))
        best_need = torch.where(better, need, best_need)
        best_vn = torch.where(better, vn, best_vn)
        best_n = torch.where(better[:, None], n, best_n)
        best_mu = torch.where(better, torch.exp(P[:, 14]), best_mu)
        best_hit = best_hit | hit
    return best_vn, best_need, best_n, best_mu, best_hit


def decode(params_pred, batch):
    """Physical density and friction per body: the given slot where given, else the prediction (detached by the
    caller where the physics layer must not train it)."""
    v = torch.where(batch["param_given"], batch["params"], params_pred)
    if "material" in batch:                                          # per material (cloth_windows.KIND_CENTRE): slot 0
        mat = batch["material"]                                      # is w, lam or rho, slot 3 mu, both logs
        ce = torch.as_tensor(KIND_CENTRE, device=v.device)[mat]
        sc = torch.as_tensor(KIND_SCALE, device=v.device)[mat]
        raw = torch.exp(v * sc + ce)
        return raw[:, 0], raw[:, 3]
    ce, sc = torch.as_tensor(C_CENTRE, device=params_pred.device), torch.as_tensor(C_SCALE, device=params_pred.device)
    raw = torch.exp(v * sc + ce)
    return raw[:, 0], raw[:, 3]


def build_cloth_model(A, **kw):
    """The cloth network a checkpoint's (or the trainer's) arguments A describe: ClothSimulator, or with
    A["arch"] == "qwen" lsim.model.cloth_qwen.ClothQwen, "qwen3d" ClothQwen3D (cloth, rope, soft). kw overrides (e.g.
    ckpt, bf16)."""
    assert abs(float(A.get("fps", 30.0)) * DT - 1.0) < 1e-9, (f"a {A.get('fps', 30.0):g} fps checkpoint with the layer at "
                                                              f"{1 / DT:g} fps: set LSIM_FPS")
    common = dict(prob_slots=A.get("prob_slots", False), unc=A.get("unc", False), accel=A.get("accel", False),
                  layer=A.get("layer", 1))
    if A.get("fp32"):                                                # --fp32: no bf16 autocast (precision test)
        common["bf16"] = False
    if A.get("oracle_v"):                                            # --oracle_v: true velocities as inputs (diagnostic)
        assert A.get("arch") == "qwen", "--oracle_v: the Qwen cloth network only"
        common["oracle_v"] = True
    if A.get("patch"):                                               # --patch R: the grid stencil in the vertex's frame
        assert A.get("arch") == "qwen", "--patch: the Qwen cloth network only"
        common["patch"] = int(A["patch"])
    if A.get("own"):                                                 # --own: cloth tokens in their own frames
        assert A.get("arch") == "qwen", "--own: the Qwen cloth network only"
        common["own"] = True
        if A.get("own_up"):                                          # --own_up: + the world-z parts and a z force
            common["own_up"] = True
    restore = kw.pop("restore", None)
    common.update(kw)
    m = _build(A, common)
    # shape restoration in the exact layer (off unless the checkpoint trained with --restore, or the caller / the
    # environment variable LSIM_RESTORE asks for it at inference: "1" the measured bands, or a restore_config spec)
    import os
    spec = restore if restore is not None else os.environ.get("LSIM_RESTORE", A.get("restore", ""))
    m.restore = restore_config(spec)
    m.slot_detach = bool(A.get("side_apart", False))                  # --side_apart (cloth_train; Qwen networks)
    m.drop_fine = bool(A.get("drop_fine", False))                     # --drop_fine (cloth_train; Qwen networks)
    m.direct = bool(A.get("direct", False))                           # --direct: no exact layer (the plain transformer)
    m.chunk = int(A.get("chunk", 0) or 0)                              # --chunk K: K frames per call (cloth_train)
    return m


def _build(A, common):
    if A.get("noise_dim", 0):                                        # generative (cloth_qwen: conditional RMSNorm)
        assert A.get("arch", "graph") in ("qwen", "qwen3d"), "the noise input exists for the Qwen networks only"
        common["noise_dim"] = int(A["noise_dim"])
    if A.get("arch", "graph") == "qwen3d":
        from lsim.model.cloth_qwen import ClothQwen3D
        return ClothQwen3D(A.get("d", 256), A.get("layers", 8), heads=A.get("heads", 4), kv_heads=A.get("kv_heads", 2),
                           mlp_hidden=A.get("mlp") or 3 * A.get("d", 256), **common)
    if A.get("arch", "graph") == "qwen":
        from lsim.model.cloth_qwen import ClothQwen
        return ClothQwen(A.get("d", 256), A.get("layers", 8), heads=A.get("heads", 4), kv_heads=A.get("kv_heads", 2),
                         mlp_hidden=A.get("mlp") or 3 * A.get("d", 256), **common)
    return ClothSimulator(A.get("d", 128), A.get("layers", 6), global_attn=not A.get("no_global", True),
                          direct=A.get("direct", False), **common)
