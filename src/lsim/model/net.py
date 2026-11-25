"""The learned simulator's network (DIRECTION.md "Network"; first version).

Tokens are (vertex, frame) pairs over a window of P given + K chunk frames. Each block:
  time    each vertex attends to its own earlier frames (causal; absent past frames masked);
  space   message passing over mesh-neighbour edges and contact edges (different bodies, < 3 cm,
          re-found every frame from that frame's positions by spatial hashing);
  object  vertices pool into their body's token (with its parameter slots) and read it back.
Heads: a force on every contact edge applied equal and opposite (f_ij = (g_ij - g_ji) / 2), so
momentum is conserved by construction -- except edges touching the gripper, whose force arrives
through the external force slot instead (given, or EMPTY and predicted by the external-force
head); a clean value for every parameter slot (used where the slot is hidden). The exact physics
layer (physics.py) turns per-vertex forces into rigid-body and joint motion with the slot values
(given or predicted); the gripper follows its recorded path. With direct=True the network is
baseline B2 instead: one acceleration per vertex, no physics layer.
"""
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as Fn

from lsim.train.windows import CENTRE, SCALE

R_CONTACT = 0.03
DT = 1.0 / 30.0


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def rbf(d, n=8, r=R_CONTACT):
    """Radial basis expansion with a smooth polynomial cutoff (DimeNet): no jump at the radius."""
    mu = torch.linspace(0, r, n, device=d.device)
    x = (d / r).clamp(max=1.0)
    env = 1 - 6 * x ** 5 + 15 * x ** 4 - 10 * x ** 3
    return torch.exp(-((d[:, None] - mu) / (r / n)) ** 2) * env[:, None]


def contact_edges(pos, body, sample, active, r=R_CONTACT):
    """Contact pairs for every frame at once, by spatial hashing. Tokens (vertex, frame) are binned
    into cells of size r and paired with the tokens in the 27 neighbouring cells of the same
    sample and frame. Returns token pairs (2, E), token = vertex * Fw + frame, both directions,
    different bodies, at least one endpoint active, |x_i - x_j| < r. (A cdist per sample and
    frame took 290 ms for 4 windows; this takes 19 ms and finds the same pairs.)"""
    N, Fw, _ = pos.shape
    dev = pos.device
    x = pos.reshape(-1, 3)
    tok = torch.arange(N * Fw, device=dev)
    v, f = tok // Fw, tok % Fw
    cell = torch.floor(x / r).long()
    cell = cell - cell.min(0).values + 1                              # >= 1, so offsets stay >= 0
    span = cell.max(0).values + 2

    def key_of(c):
        return (((sample[v] * Fw + f) * span[0] + c[:, 0]) * span[1] + c[:, 1]) * span[2] + c[:, 2]
    order = torch.argsort(key_of(cell))
    skey = key_of(cell)[order]
    act = active[v]
    src = tok[act]
    I, J = [], []
    for o in torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=dev):
        k = key_of(cell + o)[act]
        lo = torch.searchsorted(skey, k, side="left")
        cnt = torch.searchsorted(skey, k, side="right") - lo
        i = torch.repeat_interleave(src, cnt)
        j = order[torch.repeat_interleave(lo - torch.cumsum(cnt, 0) + cnt, cnt) + torch.arange(len(i), device=dev)]
        keep = (body[v[i]] != body[v[j]]) & ((x[i] - x[j]).norm(dim=-1) < r)
        I.append(i[keep]); J.append(j[keep])
    e = torch.stack([torch.cat(I), torch.cat(J)])
    return torch.unique(torch.cat([e, e.flip(0)], 1), dim=1)


class Block(nn.Module):
    def __init__(self, d, heads=4, causal=True):
        super().__init__()
        self.causal = causal
        self.ln_t, self.ln_s, self.ln_o = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.edge = mlp(2 * d + 16, d, d)
        self.node = mlp(2 * d, d, d)
        self.obj_in = mlp(d, d, d)
        self.obj_out = mlp(2 * d, d, d)
        self.ff = mlp(d, 4 * d, d)
        self.ln_f = nn.LayerNorm(d)

    def forward(self, h, Fw, e_idx, e_feat, tok_body, n_body_tok, body_tok, absent):
        N = h.shape[0] // Fw
        # time: (N, Fw, d), causal over frames; absent past frames are never attended to
        x = self.ln_t(h).view(N, Fw, -1)
        causal = torch.triu(torch.ones(Fw, Fw, dtype=torch.bool, device=h.device), 1) if self.causal else \
            torch.zeros(Fw, Fw, dtype=torch.bool, device=h.device)
        mask = causal[None] | absent[:, None, :]                      # (N, Fw, Fw), True = blocked
        mask = mask & ~torch.eye(Fw, dtype=torch.bool, device=h.device)[None]   # a token always sees itself
        mask = mask.repeat_interleave(self.attn.num_heads, 0)
        h = h + self.attn(x, x, x, attn_mask=mask, need_weights=False)[0].reshape(N * Fw, -1)
        # space: message passing over this frame's edges
        x = self.ln_s(h)
        src, dst = e_idx
        m = self.edge(torch.cat([x[src], x[dst], e_feat.to(x.dtype)], -1))
        # scatter-adds in fp32: bf16 index_add_ has no native atomics and took 42% of the step
        agg = torch.zeros(x.shape, device=x.device, dtype=torch.float32).index_add_(0, dst, m.float())
        h = h + self.node(torch.cat([x, agg.to(x.dtype)], -1))
        # object: pool per (body, frame) token, combine with the body's slot embedding, broadcast
        x = self.ln_o(h)
        oi = self.obj_in(x).float()
        pooled = torch.zeros(n_body_tok, x.shape[1], device=h.device).index_add_(0, tok_body, oi)
        cnt = torch.zeros(n_body_tok, device=h.device).index_add_(0, tok_body, torch.ones_like(tok_body, dtype=torch.float32))
        pooled = pooled / cnt.clamp_min(1)[:, None]
        h = h + self.obj_out(torch.cat([x, (pooled + body_tok)[tok_body].to(x.dtype)], -1))
        return h + self.ff(self.ln_f(h))


def near_features(pos, vel, body, role, normal, bounds, r):
    """What a dynamic vertex has near it, as token inputs (N, Fw, 34): its nearest STATIC point (offset, distance,
    height along that point's normal, the normal, found) and its nearest point of ANOTHER moving body or the gripper
    (offset, distance, relative velocity, is-gripper, found), within r, each with a radial expansion of the distance.
    By cdist per sample (a few thousand nodes); zero beyond r and for vertices that are not dynamic."""
    N, Fw, _ = pos.shape
    out = pos.new_zeros(N, Fw, 34)
    for a, b in bounds:
        ar = torch.arange(a, b, device=pos.device)
        q = ar[role[a:b] == 2]
        if len(q) == 0:
            continue
        xq = pos[q].transpose(0, 1)                                     # (Fw, nq, 3)
        col = 0
        for static in (True, False):
            t = ar[(role[a:b] == 0) if static else (role[a:b] > 0)]
            if len(t):
                dm = torch.cdist(xq, pos[t].transpose(0, 1))            # (Fw, nq, nt)
                if not static:
                    dm = dm.masked_fill((body[q][:, None] == body[t][None, :])[None], float("inf"))
                dist, j = dm.min(-1)                                    # (Fw, nq)
                dist, j = dist.T, t[j.T]                                # (nq, Fw), global vertex index
                fr = torch.arange(Fw, device=pos.device)[None].expand_as(j)
                off = pos[j, fr] - pos[q]
                ok = (dist < r).float()[..., None]
                d1 = dist.clamp(max=r)
                f = [off / r, d1[..., None] / r, rbf(d1.reshape(-1), r=r).view(len(q), Fw, 8)]
                if static:
                    nrm = normal[j]
                    f += [-(off * nrm).sum(-1, keepdim=True) / r, nrm]
                else:
                    f += [vel[j, fr] - vel[q], (role[j] == 1).float()[..., None]]
                out[q, :, col:col + 17] = torch.cat(f + [torch.ones_like(ok)], -1) * ok
            col += 17
    return out


class Simulator(nn.Module):
    def __init__(self, d=128, layers=6, n_params=11, ckpt=True, bf16=True, causal=True, direct=False, pair_norm=False,
                 support=False, r_contact=R_CONTACT, grip_force=False, trunk="graph", n_heads=4, kv_heads=2, r_see=0.10,
                 touch_cv=False):
        """direct: baseline B2 -- the same network predicting each vertex's acceleration, with no
        physics layer (DIRECTION.md "Baselines"); the force heads are then unused.
        pair_norm: a contact edge's force is the head's output x g x the reduced mass of the two
        BODIES / the number of contact pairs between them in that frame, so the bodies' total
        interaction is m_red g x the mean output (a resting body: mean 1 upward). Without it
        (per-vertex reduced masses) holding a body still needs the outputs to sum to its vertex
        count over however many edges touch -- a reciprocal of the contact fraction the network
        can only approximate (0.9 mm one-step bias at rest, 31 mm after 30 frames, overfit100).
        support: a rigid dynamic body touching static surfaces (table, floor, ramp, cabinet walls) is
        held up by default -- m g upward, shared over its static contact pairs by closeness (a
        pair counts fully below 1 cm, not at all beyond 2 cm; the full weight from 3 such pairs) --
        so a zero output keeps a resting body at rest, as constant velocity does for B0/B2, while
        free flight stays exactly ballistic. The total contact force on ANY body at rest is m g
        upward whatever the surface's tilt; the network learns the rest (friction, impacts,
        tipping, stacks). It also holds a body on a slope that should slide (model v0: the Physics
        101 cube never moved), hence
        support="coulomb": the default is Coulomb contact against the static surfaces -- a normal
        force against gravity and the given pushes (surface normal from the static points' normals),
        and friction that would stop the body within one frame, capped at mu N with the pair's
        friction the average of the object's slot and the surface's (Kamino's rule). A body rests,
        slides down a slope at g (sin - mu cos) or slides to a stop on the table by default.
        r_contact: the radius of a contact edge (m). The network sees another body, and can push on it, only
        through these edges, built at the START of a frame: at the default 3 cm a body that travels further than
        that in a frame (v0: up to 6 cm) is hit or lands unseen (the one-episode test, 2026-11-23). Windows must
        be built with make_window(near=r_contact) so the static points reach as far.
        grip_force: a learned force on contact edges that touch the gripper too (by default the gripper acts
        only through the external-force slot, so the network cannot correct a given gripper force).
        trunk="attn" (2026-11-23, the one-episode test: the graph sees and pushes only through 3 cm edges): the
        cloth network's blocks (cloth_qwen.QwenBlock) in place of message passing -- every node of a sample attends
        to every node of it in the same frame (rotary positions on the world position, cm), then to its own frames;
        each dynamic vertex is given what is nearest to it within r_see as inputs (near_features); and a per-vertex
        force head (units of the body's weight / its vertices, zero at init) is added to the edge forces, so the
        network can act on a body with no contact edge. Windows are built with make_window(near=self.near).
        touch_cv (2026-11-23, DIRECTION.md "The rigid layer model starts behind constant velocity"): a rigid body
        with a contact edge to the gripper or to another moving body in a frame gets NO default in that frame --
        no support, gravity cancelled through its centre of mass, the given gripper force cancelled where it
        acts -- so a zero output there is constant velocity (B2's starting point) and the network's forces are the
        body's whole net force; a body in flight or alone on fixed surfaces keeps the exact layer. The layer alone
        is 1.74 of constant velocity on held-out moving bodies (the recorded gripper force integrated with nothing
        to balance it), this switch 0.90 before any learning. The force slot is then read, not integrated."""
        super().__init__()
        self.d, self.ckpt, self.bf16, self.n_params, self.direct = d, ckpt, bf16, n_params, direct
        self.pair_norm, self.support = pair_norm, support
        self.r_contact, self.grip_force, self.touch_cv = r_contact, grip_force, touch_cv
        self.trunk, self.r_see = trunk, r_see
        self.near = r_see if trunk == "attn" else r_contact          # how far from moving things static points are kept
        # vel, rest, given external force, role, force-slot state, given / present frame, tau,
        # static surface material (log mu, logit e; zero for bodies)
        self.vert_in = mlp(3 + 3 + 3 + 3 + 3 + 2 + 1 + 2 + (34 if trunk == "attn" else 0), d, d)
        self.edge_in = mlp(3 + 1 + 8 + 3 + 1, 32, 16)                 # offset, dist, rbf, rel vel, is-contact
        self.body_in = mlp(2 * n_params + 4, d, d)
        if trunk == "attn":
            from lsim.model.cloth_qwen import QwenBlock, RMSNorm        # here: cloth_qwen imports this module
            self.dh = d // n_heads
            self.blocks = nn.ModuleList([QwenBlock(d, n_heads, kv_heads, 3 * d, layers) for _ in range(layers)])
            self.out_norm = RMSNorm(d)
            self.vforce = mlp(d, d, 3)                                # per dynamic vertex, units of body weight / vertices
        else:
            self.blocks = nn.ModuleList([Block(d, causal=causal) for _ in range(layers)])
        self.force = mlp(2 * d + 16, d, 3)                            # contact edges
        self.ext = mlp(d, d, 3)                                       # external force on EMPTY vertices
        self.param = mlp(2 * d, d, n_params)                          # clean value of every slot
        heads = [self.force, self.ext] + ([self.vforce] if trunk == "attn" else [])
        if direct:
            self.acc = mlp(d, d, 3)                                   # B2 only: vertex acceleration, units of g
            heads.append(self.acc)
        for head in heads:                                            # start from free fall (B2: constant velocity)
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _com_world(batch, pos, params_pred):
        """Every body's centre of mass in every frame (nb, Fw, 3): its pose from its vertices
        (Kabsch per body and frame) and its CoM slot (given, or decoded where hidden)."""
        N, Fw, _ = pos.shape
        nb, body, vl = int(batch["n_bodies"]), batch["body_of"], batch["vert_local"]
        cnt = torch.zeros(nb, device=pos.device).index_add_(0, body, torch.ones(N, device=pos.device)).clamp_min(1)
        cl = torch.zeros(nb, 3, device=pos.device).index_add_(0, body, vl) / cnt[:, None]
        cx = torch.zeros(nb, Fw, 3, device=pos.device).index_add_(0, body, pos) / cnt[:, None, None]
        H = torch.zeros(nb, Fw, 3, 3, device=pos.device).index_add_(
            0, body, (vl - cl[body])[:, None, :, None] * (pos - cx[body])[:, :, None, :])
        U, _, Vt = torch.linalg.svd(H)
        V, Ut = Vt.transpose(-1, -2), U.transpose(-1, -2)
        dt = torch.sign(torch.linalg.det(V @ Ut))
        R = V @ torch.diag_embed(torch.stack([torch.ones_like(dt), torch.ones_like(dt), dt], -1)) @ Ut
        ce, sc = torch.as_tensor(CENTRE, device=pos.device), torch.as_tensor(SCALE, device=pos.device)
        rigid = (batch["body_role"] == 2) & (batch["body_type"] == 0)
        com_l = torch.where((rigid & ~batch["param_given"][:, 3])[:, None],
                            (params_pred * sc + ce)[:, 3:6] * batch["body_size"][:, None], batch["body_com"])
        return cx + (R @ (com_l - cl)[:, None, :, None])[..., 0]

    def forward(self, batch, slot_mass):
        """slot_mass(params_pred) -> per-body mass (given value where the slot is given, else
        decoded from the prediction). Returns contact forces and predicted external forces per
        vertex (N, Fw, 3), and the parameter predictions (n_bodies, n_params)."""
        pos, vel = batch["pos"], batch["vel"]                          # (N, Fw, 3)
        N, Fw, _ = pos.shape
        rest = batch["rest"]
        role_t = Fn.one_hot(batch["role"], 3).float()
        state = batch["f_state"]                                        # (N, Fw) 0 zero, 1 given, 2 empty
        f_given = torch.asinh(batch["f_ext"] * (state == 1)[..., None] / 0.1)   # N; compressed
        given = batch["frame_given"].float()                           # (Fw,) 1 for given frames
        present = (~batch["absent"]).float()                           # (N, Fw)
        feats = torch.cat([vel * present[..., None], rest[:, None].expand(-1, Fw, -1), f_given,
                           role_t[:, None].expand(-1, Fw, -1), Fn.one_hot(state.long(), 3).float(),
                           given[None, :, None].expand(N, -1, -1), present[..., None],
                           batch["tau"][None, :, None].expand(N, -1, -1), batch["surface"][:, None].expand(-1, Fw, -1)], -1)
        if self.trunk == "attn":
            from lsim.model.cloth_qwen import rope_angles, rope_sections, sample_bounds
            bounds = sample_bounds(batch)
            feats = torch.cat([feats, near_features(pos, vel, batch["body_of"], batch["role"], batch["normal"], bounds, self.r_see)], -1)
        h = self.vert_in(feats).reshape(N * Fw, -1)
        # graph over all frames: token id = vertex * Fw + frame; nothing touches an absent frame
        x, vt = pos.reshape(-1, 3), vel.reshape(-1, 3)
        ec = contact_edges(pos, batch["body_of"], batch["sample_of"], batch["role"] > 0, r=self.r_contact)
        em = batch["edges"].T
        em = torch.cat([em, em.flip(0)], 1)
        em = (em[:, :, None] * Fw + torch.arange(Fw, device=pos.device)).reshape(2, -1)
        gone = batch["absent"].reshape(-1)
        e_all, f_all, c_all = [], [], []
        for e, is_c in ((em, 0.0), (ec, 1.0)):
            e = e[:, ~(gone[e[0]] | gone[e[1]])]                      # may be empty (nothing in contact)
            off = x[e[0]] - x[e[1]]
            dist = torch.linalg.norm(off, dim=-1)
            ef = torch.cat([off / self.r_contact, dist[:, None] / self.r_contact, rbf(dist, r=self.r_contact), vt[e[0]] - vt[e[1]],
                            torch.full_like(dist[:, None], is_c)], -1)
            e_all.append(e); f_all.append(ef)
            c_all.append(torch.full((e.shape[1],), is_c, device=pos.device))
        e_idx = torch.cat(e_all, 1)
        e_feat = self.edge_in(torch.cat(f_all, 0))
        nb = int(batch["n_bodies"])
        tok_body = (batch["body_of"][:, None] * Fw + torch.arange(Fw, device=pos.device)[None]).reshape(-1)
        body_tok = self.body_in(torch.cat([batch["params"] * batch["param_given"], batch["param_given"].float(),
                                           Fn.one_hot(batch["body_type"], 4).float()], -1))
        body_tok = body_tok[:, None].expand(-1, Fw, -1).reshape(nb * Fw, -1)
        absent = batch["absent"]
        if self.trunk == "attn":
            p = (pos / 0.01).reshape(-1, 3)                              # cm, token = vertex * Fw + frame
            rs = []
            for k, D in enumerate(rope_sections(self.dh)):
                rs += rope_angles(p[:, k], D, 100.0)
            rs = tuple(rs)
            rt = rope_angles(torch.arange(N * Fw, device=pos.device) % Fw, self.dh, 100.0)
            causal = torch.tril(torch.ones(Fw, Fw, dtype=torch.bool, device=pos.device))
            t_mask = ((causal[None] & ~absent[:, None, :]) | torch.eye(Fw, dtype=torch.bool, device=pos.device)[None])[:, None]
            h = h + body_tok[tok_body]                                  # the body's slots on every token of it
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            for blk in self.blocks:
                if self.trunk == "attn":
                    h = (checkpoint(blk, h, Fw, bounds, rs, rt, t_mask, None, None, use_reentrant=False)
                         if self.training and self.ckpt else blk(h, Fw, bounds, rs, rt, t_mask, None, None))
                elif self.training and self.ckpt:            # recompute in backward: memory ~ one block
                    h = checkpoint(blk, h, Fw, e_idx, e_feat, tok_body, nb * Fw, body_tok, absent, use_reentrant=False)
                else:
                    h = blk(h, Fw, e_idx, e_feat, tok_body, nb * Fw, body_tok, absent)
        h = h.float()
        if self.trunk == "attn":
            h = self.out_norm(h)
        # parameters: each body's present GIVEN-frame tokens pooled (never the chunk being
        # predicted: a rollout has only the past), with its own slot embedding
        w_tok = (present * given[None, :]).reshape(-1)
        pooled = torch.zeros(nb, self.d, device=h.device).index_add_(0, batch["body_of"].repeat_interleave(Fw), h * w_tok[:, None])
        cnt = torch.zeros(nb, device=h.device).index_add_(0, batch["body_of"].repeat_interleave(Fw), w_tok)
        params_pred = self.param(torch.cat([pooled / cnt.clamp_min(1)[:, None], body_tok.view(nb, Fw, -1)[:, 0]], -1))
        m_body = slot_mass(params_pred)                                 # (nb,)
        # vertex mass share; static and kinematic vertices act as infinitely heavy
        cntv = torch.bincount(batch["body_of"], minlength=nb).clamp_min(1).float()
        mv = torch.where(batch["role"] == 2, m_body[batch["body_of"]] / cntv[batch["body_of"]],
                         torch.full_like(cntv[batch["body_of"]], 1e6)).repeat_interleave(Fw)   # token order: vertex * Fw + frame
        # forces on contact edges, antisymmetric; none on edges touching the gripper (its force is
        # the external slot). The head predicts an acceleration-like quantity (units of g) scaled by
        # the pair's reduced mass (raw forces of +-30 N on a 5 g object diverged at initialisation).
        kin = (batch["role"] == 1).repeat_interleave(Fw)
        src, dst = e_idx
        is_c = torch.cat(c_all) if self.grip_force else torch.cat(c_all) * ~(kin[src] | kin[dst])
        touched = None
        if self.touch_cv:                                              # (body, frame): the gripper or a moving body within reach
            rol = batch["role"].repeat_interleave(Fw)
            near_mover = (torch.cat(c_all) > 0) & (rol[src] == 2) & (rol[dst] > 0)
            touched = torch.zeros(nb * Fw, device=pos.device).index_add_(
                0, batch["body_of"].repeat_interleave(Fw)[src] * Fw + src % Fw, near_mover.float()) > 0
        if self.pair_norm:
            mb = torch.where(batch["body_role"] == 2, m_body, torch.full_like(m_body, 1e6))
            bt = batch["body_of"].repeat_interleave(Fw)                # token -> body
            ba, bb = bt[src], bt[dst]
            key = (ba * nb + bb) * Fw + src % Fw                       # (body, body, frame)
            n_dir = torch.zeros(nb * nb * Fw, device=pos.device).index_add_(0, key, is_c.float())
            mu_red = mb[ba] * mb[bb] / (mb[ba] + mb[bb]) / n_dir[key].clamp_min(1.0)   # pairs A-B: one edge each way
        else:
            mu_red = mv[src] * mv[dst] / (mv[src] + mv[dst])
        g = self.force(torch.cat([h[src], h[dst], e_feat], -1)) * is_c[:, None] * (9.81 * mu_red[:, None])
        # pair (i->j) with (j->i): edges were built symmetric, so average with the reverse via scatter
        f_vert = torch.zeros(N * Fw, 3, device=pos.device).index_add_(0, src, g) \
            - torch.zeros(N * Fw, 3, device=pos.device).index_add_(0, dst, g)
        if self.support:
            rid = ((batch["role"] == 2) & (batch["body_type"][batch["body_of"]] == 0)).repeat_interleave(Fw)
            stat = (batch["role"] == 0).repeat_interleave(Fw)
            sel = (is_c > 0) & rid[src] & stat[dst]                     # rigid dynamic -> static, one direction
            d = torch.linalg.norm(x[src] - x[dst], dim=-1)
            w0 = ((0.02 - d) / 0.01).clamp(0.0, 1.0) * sel
            bt = batch["body_of"].repeat_interleave(Fw)
            bkey = bt[src] * Fw + src % Fw                             # (body, frame)
            W = torch.zeros(nb * Fw, device=pos.device).index_add_(0, bkey, w0)
            gate = (W / 3.0).clamp(max=1.0)                            # the full force from 3 close pairs
            if touched is not None:
                gate = gate * (~touched).float()
            pd = params_pred.detach()
            com = self._com_world(batch, pos, pd).reshape(nb * Fw, 3)
            mbf = m_body.repeat_interleave(Fw)[:, None]
            up = torch.tensor([0.0, 0.0, 9.81], device=pos.device)     # -g
            if self.support == "coulomb":
                Wc = W.clamp_min(1e-9)
                nrm = batch["normal"].repeat_interleave(Fw, 0)
                nsum = torch.zeros(nb * Fw, 3, device=pos.device).index_add_(0, bkey, w0[:, None] * nrm[dst])
                nh = nsum / nsum.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                mu_s = torch.zeros(nb * Fw, device=pos.device).index_add_(
                    0, bkey, w0 * torch.exp(batch["surface"].repeat_interleave(Fw, 0)[dst, 0])) / Wc
                ce, sc = torch.as_tensor(CENTRE, device=pos.device), torch.as_tensor(SCALE, device=pos.device)
                pv = torch.where(batch["param_given"], batch["params"], pd)
                mu = 0.5 * (torch.exp((pv * sc + ce)[:, 1]).repeat_interleave(Fw) + mu_s)   # Kamino: the average
                tok = torch.arange(N * Fw, device=pos.device)
                fg = (batch["f_ext"] * (batch["f_state"] == 1)[..., None]).reshape(N * Fw, 3)
                Fx = torch.zeros(nb * Fw, 3, device=pos.device).index_add_(0, bt * Fw + tok % Fw, fg)
                v = (com - self._com_world(batch, pos - vel * DT, pd).reshape(nb * Fw, 3)) / DT
                hold = mbf * up - Fx                                   # what the surface must supply against gravity + pushes
                N_ = (hold * nh).sum(-1).clamp_min(0.0)                # pushes only, never pulls
                req = hold - mbf * v / DT                              # ... and to stop the body within one frame
                Ft = req - (req * nh).sum(-1, keepdim=True) * nh
                Ft = Ft * (mu * N_ / Ft.norm(dim=-1).clamp_min(1e-9)).clamp(max=1.0)[:, None]   # Coulomb: |f| <= mu N
                Fdef = (N_[:, None] * nh + Ft) * gate[:, None]
            else:
                Fdef = mbf * up * gate[:, None]
            # share it over the pairs so that the line of action passes through the centre of mass
            # (zero torque) where the contacts allow it: linear reweighting in the plane normal to it
            u = Fdef / Fdef.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            u = torch.where(Fdef.norm(dim=-1, keepdim=True) > 1e-9, u, torch.tensor([0.0, 0.0, 1.0], device=pos.device))
            ax_ = torch.where((u[:, :1].abs() < 0.9), torch.tensor([1.0, 0.0, 0.0], device=pos.device),
                              torch.tensor([0.0, 1.0, 0.0], device=pos.device))
            e1 = torch.cross(u, ax_, dim=-1); e1 = e1 / e1.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            E = torch.stack([e1, torch.cross(u, e1, dim=-1)], 1)        # (nb * Fw, 2, 3)
            pe = (E[bkey] @ x[src][..., None])[..., 0]
            cw = (E @ com[..., None])[..., 0]
            Wc = W.clamp_min(1e-9)
            pbar = torch.zeros(nb * Fw, 2, device=pos.device).index_add_(0, bkey, w0[:, None] * pe) / Wc[:, None]
            dp = pe - pbar[bkey]
            C = torch.zeros(nb * Fw, 2, 2, device=pos.device).index_add_(
                0, bkey, w0[:, None, None] * dp[:, :, None] * dp[:, None, :]) / Wc[:, None, None]
            lam = torch.linalg.solve(C + 1e-6 * torch.eye(2, device=pos.device), (cw - pbar)[..., None])[..., 0]
            w = w0 * (1.0 + (lam[bkey] * dp).sum(-1)).clamp_min(0.0)
            Wn = torch.zeros(nb * Fw, device=pos.device).index_add_(0, bkey, w)
            share = w / Wn[bkey].clamp_min(1e-9)
            f_vert += 2.0 * torch.zeros(N * Fw, 3, device=pos.device).index_add_(0, src, share[:, None] * Fdef[bkey])   # x 2: f_vert is halved below
        # external force on empty vertices, in units of the body's weight
        f_ext = self.ext(h).view(N, Fw, 3) * (9.81 * m_body[batch["body_of"]])[:, None, None]
        if self.direct:                                               # B2: f_contact slot carries accelerations
            return 9.81 * self.acc(h).view(N, Fw, 3), f_ext, params_pred
        f_out = 0.5 * f_vert.view(N, Fw, 3)
        if touched is not None:                                        # no default on a touched rigid body: cancel gravity
            bo = batch["body_of"]                                      # through its centre of mass and the given force
            tb = touched.view(nb, Fw) & ((batch["body_role"] == 2) & (batch["body_type"] == 0))[:, None]
            com = self._com_world(batch, pos, params_pred.detach())    # (nb, Fw, 3)
            rbar = torch.zeros(nb, Fw, 3, device=pos.device).index_add_(0, bo, pos) / cntv[:, None, None]
            dr = pos - rbar[bo]
            Sm = torch.zeros(nb, Fw, 3, 3, device=pos.device).index_add_(0, bo, dr[..., :, None] * dr[..., None, :])
            reg = (1e-4 * Sm.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12))[..., None, None] * torch.eye(3, device=pos.device)
            lam = torch.linalg.solve(Sm + reg, (com - rbar)[..., None])[..., 0]
            w = 1.0 / cntv[bo][:, None] + (dr * lam[bo]).sum(-1)       # sum 1, weighted mean position = the centre of mass
            anti = w[..., None] * (9.81 * m_body)[bo][:, None, None] * torch.tensor([0.0, 0.0, 1.0], device=pos.device)
            f_out = f_out + (anti - batch["f_ext"] * (state == 1)[..., None]) * tb[bo][..., None]
        if self.trunk == "attn":                                       # a handle on every dynamic vertex, edge or not
            fv = self.vforce(h).view(N, Fw, 3) * ((9.81 * m_body / cntv)[batch["body_of"]] * (batch["role"] == 2))[:, None, None]
            return f_out + fv, f_ext, params_pred
        return f_out, f_ext, params_pred


def build_model(args, n_params):
    """The network a checkpoint's (or the trainer's) args describe: the model (Simulator), B2
    (Simulator, direct) or B1 (lsim.model.gns.GNS)."""
    if args.get("arch", "net") == "gns":
        from lsim.model.gns import GNS
        return GNS(n_params=n_params)
    return Simulator(args.get("d", 128), args.get("layers", 6), n_params=n_params, direct=args.get("direct", False),
                     pair_norm=args.get("pair_norm", False),
                     support="coulomb" if args.get("coulomb") else args.get("support", False),
                     r_contact=args.get("r_contact", R_CONTACT), grip_force=args.get("grip_force", False),
                     trunk=args.get("trunk", "graph"), n_heads=args.get("n_heads", 4), kv_heads=args.get("kv_heads", 2),
                     touch_cv=args.get("touch_cv", False))
