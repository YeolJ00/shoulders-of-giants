"""The cloth network as a Qwen3-style Transformer (2026-10-26; the user: "big designs to follow Qwen", ~500M).

A drop-in for lsim.model.cloth_net.ClothSimulator: the same batch in, the same outputs (internal and contact forces in
units of the vertex weight, the slot predictions, the contact edges), the same exact layer and losses after it. What
changes is the network between the inputs and the heads:

  tokens     one per (node, frame) -- cloth vertices, the gripper's points, static surface points -- as before
  inputs     the node's own state (as ClothSimulator) plus, for a cloth vertex, its local geometry given precisely:
             each mesh neighbour's offset, strain and relative velocity in a fixed slot by its rest direction (8
             slots, like a ViT patch on the regular grid), its nearest gripper / static point and its nearest fold
             layer (both from the same spatial hashing as before). Message passing is gone: it only carried this
             local geometry, which attention cannot resolve (strains of 1e-4) and which costs nothing as input.
  material   the cloth's slot embedding (given values and flags) added to every token of its sample
  blocks     Qwen3's: pre-RMSNorm residual sub-layers, x + Attn(RMSNorm(x)), x + SwiGLU(RMSNorm(x)); attention with
             grouped-query heads, RMSNorm on queries and keys, no biases anywhere. Each block attends in space then
             in time (divided space-time attention, as video Transformers; joint attention over 8-16 frames of a
             ~1,500-token sample would cost 16x more):
               space  every node of a sample within one frame, rotary positions on the cloth's rest coordinates
                      (u, v in cm; other nodes at 0) -- Qwen2-VL's M-RoPE (t, h, w) with the cloth's grid as the image
               time   each node over its own frames, causal, absent past frames masked, rotary positions on the frame
  heads      a final RMSNorm, then the same heads as ClothSimulator (the force heads start at zero: free fall +
             Coulomb, the exact layer's default)

Sizes (d, layers, heads, kv heads, MLP): ~8M (256, 8, 4, 2, 768), ~63M (512, 16, 8, 4, 1536), ~490M (1024, 32, 16,
8, 3072; Qwen3-0.6B's width, heads and MLP).

ClothQwen3D (end of this file; `cloth_train --arch qwen3d`, 2026-10-30) is the same network for cloth, rope and soft
bodies: 3D rest positions in the rotary (three sections of the head dim), 26 neighbour slots by 3D rest direction,
a learned material embedding. ClothQwen and its checkpoints are unchanged (tests/test_cloth_qwen.py checks them bit for
bit).

Generative (2026-11-02; `cloth_train --gen energy --noise_dim 32`): with noise_dim > 0 each forward pass draws a
Gaussian noise vector per body per frame, which enters every block through conditional RMSNorm -- RMSNorm(x) (1 + s(z))
+ b(z), s and b from a small MLP of the noise and a per-block linear map that starts at zero (adaLN-zero), so a
deterministic checkpoint loaded into it gives the same outputs bit for bit at init. One vector per body (not per
vertex) varies the whole cloth coherently; trained with the energy score, the network samples one future instead of
the mean of all of them. set_noise(model, seed, zero) fixes the draw for evaluation.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

from lsim.model.cloth_net import DT, G, R_CONTACT, R_SELF, cloth_edges
from lsim.model.net import mlp
from lsim.train.cloth_windows import KIND

N_SLOT = 8                                         # mesh neighbours by rest direction: E, NE, N, NW, W, SW, S, SE
SLOT_F = 12                                        # rest offset, offset, two strains, relative velocity, present
CONTACT_F = 13                                     # offset, distance, height over its normal, kind, surface, rel. velocity, count
ROPE_SPACE, ROPE_TIME = 100.0, 100.0               # rotary bases: positions up to ~60 cm, 16 frames
SELF_F = 8                                         # offset, distance, relative velocity, count


# The elementwise pieces fused by torch.compile (dynamic shapes: the token count changes every batch). Uncompiled,
# the 509M model spent ~13 s of a 21 s step in fp32 elementwise kernels and ~6 s waiting on their launches.
_compile = torch.compile(dynamic=True) if hasattr(torch, "compile") else (lambda f: f)


@_compile
def _rms_norm(x, w, eps: float):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


@_compile
def _rotate(x, cos, sin):
    xf = x.float()
    x1, x2 = xf[..., :x.shape[-1] // 2], xf[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


@_compile
def _swiglu(g, u):
    return Fn.silu(g) * u


@_compile
def _masked_softmax(a, mask, scale: float):
    return (a.float() * scale).masked_fill(~mask, float("-inf")).softmax(-1).to(a.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return _rms_norm(x, self.weight, self.eps)


def rope_angles(p, D, base):
    """cos, sin (T, 1, D / 2) of positions p (T,) for D rotated dimensions (rotate-half convention, as Qwen)."""
    inv = base ** (-torch.arange(0, D, 2, device=p.device, dtype=torch.float32) / D)
    ang = p.float()[:, None, None] * inv
    return ang.cos(), ang.sin()


def rotate(x, cos, sin):
    """x (T, H, D) rotated by angles of cos / sin (T, 1, D / 2)."""
    return _rotate(x, cos, sin)


class Attention(nn.Module):
    """Grouped-query attention with RMSNorm on queries and keys, no biases (Qwen3). Projections, norms and rotary
    positions run on the flat tokens; only the attention product sees the (padded) groups."""

    def __init__(self, d, heads, kv_heads, head_dim, n_layers):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv_heads, head_dim
        self.q = nn.Linear(d, heads * head_dim, bias=False)
        self.k = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.v = nn.Linear(d, kv_heads * head_dim, bias=False)
        self.o = nn.Linear(heads * head_dim, d, bias=False)
        self.q_norm, self.k_norm = RMSNorm(head_dim), RMSNorm(head_dim)
        for m in (self.q, self.k, self.v):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.o.weight, std=0.02 / math.sqrt(2 * n_layers))

    def qkv(self, x, rope):
        """x (T, d) flat tokens -> q (T, H, dh), k, v (T, H, dh) (key / value heads repeated), q and k normed and
        rotated by rope."""
        T = x.shape[0]
        q = rope(self.q_norm(self.q(x).view(T, self.h, self.dh)))
        k = rope(self.k_norm(self.k(x).view(T, self.kv, self.dh)))
        v = self.v(x).view(T, self.kv, self.dh)
        rep = self.h // self.kv
        return q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)

    def space(self, x, rope, bounds, Fw):
        """Every node of a sample attends to every node of it in the same frame. A sample's tokens are contiguous
        (node-major, then frame), so each sample is a (Fw, H, n, dh) view: no padding, no mask, the flash kernel."""
        q, k, v = self.qkv(x, rope)
        out = []
        for a, b in bounds:
            sl = [z[a * Fw:b * Fw].view(b - a, Fw, self.h, self.dh).permute(1, 2, 0, 3) for z in (q, k, v)]
            o = Fn.scaled_dot_product_attention(*sl)                     # (Fw, H, n, dh)
            out.append(o.permute(2, 0, 1, 3).reshape((b - a) * Fw, -1))
        return self.o(torch.cat(out))

    def time(self, x, rope, Fw, mask):
        """Each node attends to its own frames (mask (N, 1, Fw, Fw), True = may attend), by plain matmuls: at 8-16
        frames the fused kernels' 64-wide tiles waste most of their work."""
        q, k, v = (z.view(-1, Fw, self.h, self.dh).transpose(1, 2) for z in self.qkv(x, rope))
        a = _masked_softmax(q @ k.transpose(-1, -2), mask, self.dh ** -0.5)
        return self.o((a @ v).transpose(1, 2).reshape(x.shape[0], -1))


class SwiGLU(nn.Module):
    def __init__(self, d, hidden, n_layers):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x):
        return self.down(_swiglu(self.gate(x), self.up(x)))


class QwenBlock(nn.Module):
    """x + SpaceAttn(RMSNorm(x)); x + TimeAttn(RMSNorm(x)); x + SwiGLU(RMSNorm(x)). With noise_dim (the generative
    model): each RMSNorm conditioned on the noise, RMSNorm(x) (1 + s) + b, s and b from a zero-initialised linear map
    of the noise embedding (identity at init)."""

    def __init__(self, d, heads, kv_heads, mlp_hidden, n_layers, noise_dim=0):
        super().__init__()
        dh = d // heads
        self.n_s, self.n_t, self.n_f = RMSNorm(d), RMSNorm(d), RMSNorm(d)
        self.space = Attention(d, heads, kv_heads, dh, n_layers)
        self.time = Attention(d, heads, kv_heads, dh, n_layers)
        self.mlp = SwiGLU(d, mlp_hidden, n_layers)
        if noise_dim:
            self.mod = nn.Linear(d, 6 * d)                               # (s, b) for each of the three norms
            nn.init.zeros_(self.mod.weight); nn.init.zeros_(self.mod.bias)

    def forward(self, h, Fw, bounds, rs, rt, t_mask, zemb=None, zidx=None):
        """h (N * Fw, d), token = node * Fw + frame; bounds: each sample's node range [a, b); rs = (cos_u, sin_u,
        cos_v, sin_v), rt = (cos, sin) the space / time rotary angles per token (T, 1, .); t_mask (N, 1, Fw, Fw).
        zemb (B * Fw, d): the noise embedding per (body, frame), zidx (T,) each token's row of it (generative)."""
        m = None if zemb is None else self.mod(zemb)                     # (B * Fw, 6 d), fp32

        def nrm(norm, x, k):
            y = norm(x)
            if m is None:
                return y
            d = y.shape[-1]
            s, b = m[:, 2 * k * d:(2 * k + 1) * d][zidx], m[:, (2 * k + 1) * d:(2 * k + 2) * d][zidx]
            return (y.float() * (1.0 + s) + b).to(y.dtype)

        def rope_s(z):                                               # rs: (cos, sin) per section of the head dim
            if len(rs) == 4:                                             # (u, v): halves
                half = z.shape[-1] // 2
                return torch.cat([rotate(z[..., :half], rs[0], rs[1]), rotate(z[..., half:], rs[2], rs[3])], -1)
            parts = z.split([2 * c.shape[-1] for c in rs[0::2]], -1)     # (x, y, z): ClothQwen3D's sections
            return torch.cat([rotate(zp, c, sn) for zp, c, sn in zip(parts, rs[0::2], rs[1::2])], -1)
        h = h + self.space.space(nrm(self.n_s, h, 0), rope_s, bounds, Fw)
        h = h + self.time.time(nrm(self.n_t, h, 1), lambda z: rotate(z, *rt), Fw, t_mask)
        return h + self.mlp(nrm(self.n_f, h, 2))


def set_noise(model, seed=None, zero=False):
    """A generative model's noise for evaluation: seed (int) fixes the draws (a CUDA generator on the model's device;
    None: the global RNG), zero: no noise at all (the model's deterministic mode, for comparison). No-op for a
    deterministic model."""
    core = getattr(model, "module", model)
    if not getattr(core, "noise_dim", 0):
        return
    dev = next(core.parameters()).device
    core.gen_rng = None if seed is None else torch.Generator(device=dev).manual_seed(int(seed))
    core.gen_zero = bool(zero)


def sample_bounds(batch):
    """Each sample's node range [a, b) (collate_cloth concatenates samples, so a sample's nodes are contiguous)."""
    s = batch["sample_of"]
    assert bool((s[1:] >= s[:-1]).all()), "samples must be contiguous"
    cnt = torch.bincount(s).tolist()
    ends = [sum(cnt[:i + 1]) for i in range(len(cnt))]
    return [(e - c, e) for c, e in zip(cnt, ends) if c]


def neighbour_slots(batch, pos, vel):
    """Each cloth vertex's mesh neighbours in 8 slots by the direction of their rest offset: (N, Fw, 8 * SLOT_F)."""
    N, Fw, _ = pos.shape
    dev = pos.device
    e = batch["edges"]
    src, dst = torch.cat([e[:, 0], e[:, 1]]), torch.cat([e[:, 1], e[:, 0]])
    ro = (batch["rest"][dst] - batch["rest"][src]) * batch["size"][batch["sample_of"][src]][:, None]   # m
    sector = torch.round(torch.atan2(ro[:, 1], ro[:, 0]) / (math.pi / 4)).long() % N_SLOT
    off = pos[dst] - pos[src]                                            # (E, Fw, 3)
    rl = ro.norm(dim=-1).clamp_min(1e-9)[:, None]
    sr = off.norm(dim=-1) / rl - 1.0
    present = (~batch["absent"][src] & ~batch["absent"][dst]).float()
    f = torch.cat([(ro / 0.01)[:, None].expand(-1, Fw, -1), off / 0.01, torch.asinh(1e2 * sr)[..., None],
                   torch.asinh(1e5 * sr)[..., None], vel[dst] - vel[src], torch.ones_like(sr)[..., None]], -1)
    f = f * present[..., None]
    out = torch.zeros(N * N_SLOT, Fw, SLOT_F, device=dev)
    out[src * N_SLOT + sector] = f
    return out.view(N, N_SLOT, Fw, SLOT_F).transpose(1, 2).reshape(N, Fw, N_SLOT * SLOT_F)


def slot_relvel(batch, v):
    """Each cloth vertex's mesh neighbours' velocity relative to its own, in neighbour_slots' 8 slots: (N, Fw, 8 * 3)."""
    N, Fw, _ = v.shape
    e = batch["edges"]
    src, dst = torch.cat([e[:, 0], e[:, 1]]), torch.cat([e[:, 1], e[:, 0]])
    ro = (batch["rest"][dst] - batch["rest"][src]) * batch["size"][batch["sample_of"][src]][:, None]
    sector = torch.round(torch.atan2(ro[:, 1], ro[:, 0]) / (math.pi / 4)).long() % N_SLOT
    present = (~batch["absent"][src] & ~batch["absent"][dst]).float()
    out = torch.zeros(N * N_SLOT, Fw, 3, device=v.device)
    out[src * N_SLOT + sector] = (v[dst] - v[src]) * present[..., None]
    return out.view(N, N_SLOT, Fw, 3).transpose(1, 2).reshape(N, Fw, N_SLOT * 3)


def nearest(pairs, d, n_tok):
    """For token pairs (i, j) at distances d: a mask of each i's nearest pair, and each i's pair count."""
    dmin = torch.full((n_tok,), float("inf"), device=d.device).scatter_reduce(0, pairs[0], d, "amin")
    cnt = torch.zeros(n_tok, device=d.device).index_add_(0, pairs[0], torch.ones_like(d))
    return d <= dmin[pairs[0]], cnt


def own_frames(slots, cl):
    """Each cloth vertex's own frame per window frame (2026-11-22, `--own`), from its mesh neighbours' offsets in
    neighbour_slots' layout: e1 along the grid's u (E - W; one-sided at the border), e2 the grid's v made orthogonal to
    it, e3 their normal. (N, Fw, 3, 3) with the axes as rows (M @ a world vector = its components in the frame);
    identity for every other node, zero where the frame's neighbours are absent."""
    N, Fw, _ = slots.shape
    off = slots.view(N, Fw, -1, SLOT_F)[..., 3:6]
    u, v = off[:, :, 0] - off[:, :, 4], off[:, :, 2] - off[:, :, 6]
    e1 = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e2 = v - (v * e1).sum(-1, keepdim=True) * e1
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    M = torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -2)
    return torch.where(cl[:, None, None, None], M, torch.eye(3, device=slots.device).expand(N, Fw, 3, 3))


def contact_features(batch, pos, vel, ec, es, ret_pairs=False):
    """Each cloth token's nearest gripper / static point and nearest fold layer, with counts: (N, Fw, CONTACT_F +
    SELF_F). Tokens are vertex * Fw + frame. ret_pairs: also the (cloth token, its nearest other token) pairs."""
    N, Fw, _ = pos.shape
    n = N * Fw
    dev = pos.device
    x, vt = pos.reshape(-1, 3), vel.reshape(-1, 3)
    node = torch.arange(n, device=dev) // Fw
    cl = batch["is_cloth"][node]
    c = ec[:, cl[ec[0]] & ~cl[ec[1]]]                                   # cloth -> other
    off = x[c[0]] - x[c[1]]
    d = off.norm(dim=-1)
    near, cnt = nearest(c, d, n)
    c, off, d = c[:, near], off[near], d[near]
    j = node[c[1]]
    nd = (batch["normal"][j] * off).sum(-1)
    kj = batch["kind"][j]
    fc = torch.cat([off / R_CONTACT, d[:, None] / R_CONTACT, nd[:, None] / R_CONTACT,
                    (kj == KIND["gripper"]).float()[:, None], (kj == KIND["static"]).float()[:, None],
                    batch["surface"][j], vt[c[0]] - vt[c[1]], torch.log1p(cnt[c[0]])[:, None]], -1)
    out_c = torch.zeros(n, CONTACT_F, device=dev)
    out_c[c[0]] = fc
    s = es
    off = x[s[0]] - x[s[1]]
    d = off.norm(dim=-1)
    near, cnt = nearest(s, d, n)
    s, off, d = s[:, near], off[near], d[near]
    fs = torch.cat([off / R_SELF, d[:, None] / R_SELF, vt[s[0]] - vt[s[1]], torch.log1p(cnt[s[0]])[:, None]], -1)
    out_s = torch.zeros(n, SELF_F, device=dev)
    out_s[s[0]] = fs
    if ret_pairs:
        return torch.cat([out_c, out_s], -1).view(N, Fw, -1), c
    return torch.cat([out_c, out_s], -1).view(N, Fw, -1)


def grid_stencil(batch, R):
    """Each cloth vertex's (2R + 1)^2 grid stencil (the regular cloth grid, from its rest coordinates): vertex indices
    (N, S) with -1 outside the cloth (and for every non-cloth node), the cells' grid offsets (S, 2) as (di, dj), and
    each vertex's grid spacing h (N,) in m."""
    dev = batch["pos"].device
    N = batch["pos"].shape[0]
    cl = batch["is_cloth"]
    so = batch["sample_of"]
    rm = batch["rest"][:, :2].double() * batch["size"][so][:, None].double()       # m, centred per sample
    e = batch["edges"]
    el = (rm[e[:, 1]] - rm[e[:, 0]]).norm(dim=-1)
    S_n = int(so.max()) + 1
    h = torch.full((S_n,), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(0, so[e[:, 0]], el, "amin")
    lo = torch.full((S_n, 2), float("inf"), dtype=torch.float64, device=dev)
    lo = lo.scatter_reduce(0, so[cl][:, None].expand(-1, 2), rm[cl], "amin")
    ij = torch.round((rm - lo[so]) / h[so][:, None]).long()                        # (N, 2): i along u, j along v
    K = int(ij[cl].max()) + 2 * R + 2
    key = lambda s_, i_, j_: (s_ * K + j_ + R) * K + i_ + R
    look = torch.full((S_n * K * K,), -1, dtype=torch.long, device=dev)
    look[key(so[cl], ij[cl, 0], ij[cl, 1])] = torch.nonzero(cl).squeeze(1)
    d = torch.arange(-R, R + 1, device=dev)
    off = torch.stack(torch.meshgrid(d, d, indexing="xy"), -1).reshape(-1, 2)     # (S, 2): (di, dj), row by row in j
    ii, jj = ij[:, None, 0] + off[None, :, 0], ij[:, None, 1] + off[None, :, 1]
    inside = (ii >= -R) & (ii < K - R) & (jj >= -R) & (jj < K - R)
    idx = torch.where(inside, look[key(so[:, None], ii.clamp(-R, K - R - 1), jj.clamp(-R, K - R - 1))], -1)
    idx = torch.where(cl[:, None], idx, -1)
    return idx, off, h[so].float()


def patch_features(batch, pos, vel, idx, off, h, R):
    """The stencil in each cloth vertex's own frame (2026-11-19, `--patch R`): axes from the grid's two directions at
    the vertex (central differences, one-sided at the border) and their normal, per frame. Per cell: the deviation of
    its offset from the flat rest grid (cm), its velocity relative to the vertex (cm per frame), present; per vertex:
    its velocity (cm per frame) and gravity's direction in its frame. Returns features (N, Fw, (2R+1)^2 * 7 + 6) and
    the frames M (N, Fw, 3, 3; rows: the axes; identity for non-cloth nodes)."""
    N, Fw, _ = pos.shape
    S = idx.shape[1]
    ok = (idx >= 0)
    g_idx = idx.clamp_min(0)
    P = pos[g_idx]                                                       # (N, S, Fw, 3)
    Vv = vel[g_idx]
    c = (2 * R + 1) * R + R                                              # the centre cell
    cell = lambda di, dj: (R + dj) * (2 * R + 1) + R + di
    def side(di, dj):
        k = cell(di, dj)
        return torch.where(ok[:, k, None, None], P[:, k], pos)
    u = side(1, 0) - side(-1, 0)
    v = side(0, 1) - side(0, -1)
    e1 = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    e2 = v - (v * e1).sum(-1, keepdim=True) * e1
    e2 = e2 / e2.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    M = torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -2)           # (N, Fw, 3, 3)
    cl = batch["is_cloth"]
    eye = torch.eye(3, device=pos.device).expand(N, Fw, 3, 3)
    M = torch.where(cl[:, None, None, None], M, eye)
    rel = (P - pos[:, None]).transpose(1, 2)                             # (N, Fw, S, 3)
    rloc = torch.einsum("nfab,nfsb->nfsa", M, rel)
    flat = torch.zeros(S, 3, device=pos.device)
    flat[:, :2] = off.float()
    dev_ = (rloc - flat[None, None] * h[:, None, None, None]) / 0.01
    rv = torch.einsum("nfab,nfsb->nfsa", M, (Vv - vel[:, None]).transpose(1, 2)) * DT / 0.01
    m = ok[:, None, :, None].float()
    own = torch.einsum("nfab,nfb->nfa", M, vel) * DT / 0.01
    gl = M[..., 2] * -1.0                                                # M @ (0, 0, -1): minus the axes' z components
    f = torch.cat([(dev_ * m).reshape(N, Fw, -1), (rv * m).reshape(N, Fw, -1), m.expand(-1, Fw, -1, -1).reshape(N, Fw, -1),
                   own, gl], -1) * cl[:, None, None]
    return f, M


class ClothQwen(nn.Module):
    N_SLOTS = N_SLOT                                                 # neighbour slots per vertex (ClothQwen3D: 26)

    def __init__(self, d=256, layers=8, heads=4, kv_heads=2, mlp_hidden=768, n_params=4, ckpt=True, bf16=True,
                 prob_slots=False, unc=False, accel=False, layer=2, noise_dim=0, oracle_v=False, patch=0, own=False,
                 own_up=False):
        super().__init__()
        assert d % heads == 0 and heads % kv_heads == 0 and (d // heads) % 4 == 0
        self.d, self.ckpt, self.bf16, self.n_params, self.prob_slots, self.direct = d, ckpt, bf16, n_params, prob_slots, False
        self.layer, self.accel, self.dh = layer, accel, d // heads
        n_in = 3 + 3 + 3 + 3 + 1 + 3 + 3 + 1 + 1 + 1 + 2 + 3 + 1 + 3 + (6 if accel else 0)
        self.oracle_v = bool(oracle_v)                                   # diagnostic (cloth_train --oracle_v): the
        if oracle_v:                                                     # teacher's instantaneous velocities as inputs
            assert self.N_SLOTS == N_SLOT, "--oracle_v: the 2D cloth network only"
            n_in += 3 + 3 + 1 + N_SLOT * 3
        self.patch = int(patch)                                          # --patch R: the (2R+1)^2 grid stencil in each
        if self.patch:                                                   # vertex's own frame, and a force head in it
            assert self.N_SLOTS == N_SLOT, "--patch: the 2D cloth network only"
            n_in += (2 * self.patch + 1) ** 2 * 7 + 6
        self.own = bool(own)                                             # --own: every vector a cloth token reads and
        if self.own:                                                     # gives is in the vertex's own frame (below)
            assert self.N_SLOTS == N_SLOT and not self.patch and not oracle_v and not noise_dim, \
                "--own: the 2D cloth network, no --patch, --oracle_v or noise input"
            n_in += 3 + 10 + 6                                           # gravity; the grip; its contact's normal, move
        self.own_up = bool(own_up)                                       # --own_up: --own + the vertical, plainly: the
        if self.own_up:                                                  # world-z part of each vector a cloth token
            assert self.own, "--own_up needs --own"                      # reads (a yaw leaves it alone) and a force
            n_in += 28 + (2 if accel else 0)                             # along world z beside the own-frame one
        self.tok_in = mlp(n_in + self.N_SLOTS * SLOT_F + CONTACT_F + SELF_F, d, d)
        self.body_in = mlp(2 * n_params, d, d)
        self.blocks = nn.ModuleList([QwenBlock(d, heads, kv_heads, mlp_hidden, layers, noise_dim) for _ in range(layers)])
        self.noise_dim = noise_dim                                       # generative: noise per body per frame
        self.gen_rng, self.gen_zero = None, False                       # set_noise: a fixed draw / zero noise
        if noise_dim:
            self.noise_in = mlp(noise_dim, d, d)
        self.norm = RMSNorm(d)
        self.f_int = nn.Linear(d, 3)
        self.f_con = nn.Linear(d, 3)
        self.param = mlp(2 * d, d, 2 * n_params if prob_slots else n_params)
        self.unc = mlp(d, d, 1) if unc else None
        self.f_loc = nn.Linear(d, 3) if self.patch else None             # internal force in the vertex's own frame
        self.f_up = nn.Linear(d, 2) if self.own_up else None             # internal, contact force along world z
        for head in (self.f_int, self.f_con) + ((self.f_loc,) if self.patch else ()) + ((self.f_up,) if self.own_up else ()):   # start from the default:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)                      # free fall + Coulomb

    def _log_measure(self, batch):
        return torch.log(batch["area"].clamp_min(1e-9) / 1e-4) * batch["is_cloth"]

    def _slots(self, batch, pos, vel):
        f = neighbour_slots(batch, pos, vel)
        if getattr(self, "drop_fine", False):                           # --drop_fine: no asinh(1e5 strain)
            sh = f.shape
            f = f.view(*sh[:-1], N_SLOT, SLOT_F).clone()
            f[..., 7] = 0.0
            f = f.view(sh)
        return f

    def _body(self, batch):
        return self.body_in(torch.cat([batch["params"] * batch["param_given"], batch["param_given"].float()], -1))

    def _rope_space(self, batch, Fw):
        """Space rotary angles per token: (cos_u, sin_u, cos_v, sin_v) on the cloth's rest (u, v) in cm, each over half
        the head dim; other nodes at 0."""
        uv = batch["rest"][:, :2] * batch["size"][batch["sample_of"]][:, None] / 0.01 * batch["is_cloth"][:, None]
        uv = uv.repeat_interleave(Fw, 0)                                 # per token
        q = self.dh // 2
        return (*rope_angles(uv[:, 0], q, ROPE_SPACE), *rope_angles(uv[:, 1], q, ROPE_SPACE))

    def _forward_own(self, batch):
        """--own (2026-11-22): the same network with every vector a cloth token reads in the vertex's own frame
        (own_frames: its mesh's two grid directions and their normal, per frame) and its force given in that frame, so
        a scene and its rotated copy are the same input. The small-fit test's finding: 8 windows of one episode fit to
        0.06 of constant velocity as they are and stall at 0.71 re-yawed every step -- in world coordinates each
        orientation is a new example. A cloth token's vectors: its velocity, past acceleration and (held) coming one,
        its neighbours' offsets and relative velocities, its nearest contact's offset, relative velocity, normal and
        coming move, gravity, and the grip (the body's held vertices: where their centroid is, where it goes next,
        its coming acceleration). Other nodes (gripper, static) keep only what a yaw leaves alone: each vector's
        horizontal length and vertical part.

        --own_up (2026-11-22): on held-out cloth the --own network did not cancel gravity (16 episodes, 2,000 steps,
        one frame: vertical error 5.1 mm on held cloth where the world-frame network left 1.5 and the exact layer
        alone 5.7): 'up' is a constant in world coordinates and a learned function of the frame's tilt in its own.
        So beside each own-frame vector the token gets its world-z part (a yaw and a shift leave it alone), and the
        two forces get a component along world z from their own head (f_up)."""
        pos, vel = batch["pos"], batch["vel"]
        N, Fw, _ = pos.shape
        dev = pos.device
        cl = batch["is_cloth"]
        present = (~batch["absent"]).float()
        state = batch["f_state"]
        given = batch["frame_given"].float()
        slots = self._slots(batch, pos, vel) * cl[:, None, None]
        M = own_frames(slots, cl)                                        # (N, Fw, 3, 3)

        def own(v):                                                      # a per-token world vector, as the token reads it
            r = torch.einsum("nfab,nfb->nfa", M, v)
            flat = torch.stack([v[..., :2].norm(dim=-1), torch.zeros_like(v[..., 0]), v[..., 2]], -1)
            return torch.where(cl[:, None, None], r, flat)
        sl = slots.view(N, Fw, N_SLOT, SLOT_F)
        upz = [vel[..., 2:3] * present[..., None], sl[..., 5], sl[..., 10]]   # --own_up: the world-z parts
        rot_s = lambda a: torch.einsum("nfab,nfsb->nfsa", M, a)
        sl = torch.cat([sl[..., :3], rot_s(sl[..., 3:6]), sl[..., 6:8], rot_s(sl[..., 8:11]), sl[..., 11:]], -1)
        ec, es = cloth_edges(batch, pos)
        cf, cp = contact_features(batch, pos, vel, ec, es, ret_pairs=True)
        cf = cf * present[..., None]
        upz += [cf[..., [2, 11, 15, 19]]]
        r3 = lambda a: torch.einsum("nfab,nfb->nfa", M, a)
        cf = torch.cat([r3(cf[..., 0:3]), cf[..., 3:9], r3(cf[..., 9:12]), cf[..., 12:13],
                        r3(cf[..., 13:16]), cf[..., 16:17], r3(cf[..., 17:20]), cf[..., 20:]], -1)
        kn = batch["kin_next"]
        ex = torch.zeros(N * Fw, 6, device=dev)                          # the nearest contact's normal and coming move
        ex[cp[0]] = torch.cat([batch["normal"][cp[1] // Fw],             # (relative to the vertex's own, cm)
                               (kn.reshape(-1, 3)[cp[1]] - vel.reshape(-1, 3)[cp[0]]) * DT / 0.01], -1)
        ex = ex.view(N, Fw, 6) * present[..., None]
        upz += [ex[..., [2, 5]]]
        ex = torch.cat([r3(ex[..., :3]), r3(ex[..., 3:])], -1) * cl[:, None, None]
        # the grip: the body's held cloth vertices at each frame
        body_of = batch["body_of"]
        nb = int(batch["n_bodies"])
        key = body_of[:, None] * Fw + torch.arange(Fw, device=dev)[None]   # (N, Fw)
        hw = (batch["held"] & cl[:, None]).float()
        cnt = torch.zeros(nb * Fw, device=dev).index_add_(0, key.reshape(-1), hw.reshape(-1))
        agg = lambda v: torch.zeros(nb * Fw, 3, device=dev).index_add_(
            0, key.reshape(-1), (v * hw[..., None]).reshape(-1, 3)) / cnt.clamp_min(1)[:, None]
        xh, vh, kh = agg(pos), agg(vel), agg(kn)
        has = ((cnt > 0)[key] & cl[:, None]).float() * present
        grip = torch.cat([r3((xh[key] - pos) / 0.1), r3((kh[key] - vel) * DT / 0.01), r3((kh[key] - vh[key]) / (DT * G)),
                          torch.ones(N, Fw, 1, device=dev)], -1) * has[..., None]
        upz += [torch.stack([(xh[key] - pos)[..., 2] / 0.1, (kh[key] - vel)[..., 2] * DT / 0.01,
                             (kh[key] - vh[key])[..., 2] / (DT * G)], -1) * has[..., None],
                kn[..., 2:3], torch.asinh(batch["f_ext"][..., 2:3] * (state == 1)[..., None] / 0.01)]
        grav = -M[..., 2] * (cl[:, None, None] * present[..., None])     # M @ (0, 0, -1)
        f_given = torch.asinh(own(batch["f_ext"] * (state == 1)[..., None]) / 0.01)   # in the frame first, then asinh
        log_area = self._log_measure(batch)
        feats = [own(vel) * present[..., None], batch["rest"][:, None].expand(-1, Fw, -1),
                 Fn.one_hot(batch["role"], 3).float()[:, None].expand(-1, Fw, -1),
                 Fn.one_hot(batch["kind"], 3).float()[:, None].expand(-1, Fw, -1),
                 batch["held"].float()[..., None], f_given, Fn.one_hot(state.long(), 3).float(),
                 given[None, :, None].expand(N, -1, -1), present[..., None],
                 batch["tau"][None, :, None].expand(N, -1, -1),
                 batch["surface"][:, None].expand(-1, Fw, -1), own(batch["normal"][:, None].expand(-1, Fw, -1)),
                 log_area[:, None, None].expand(-1, Fw, -1), own(kn)]
        if self.accel:
            pv = torch.cat([vel[:, :1], vel[:, :-1]], 1)
            both = present * torch.cat([present[:, :1], present[:, :-1]], 1)
            kin = ((batch["kind"] == KIND["gripper"])[:, None] | batch["held"])[..., None]
            feats += [own((vel - pv) / (DT * G)) * both[..., None],
                      own(torch.where(kin, (kn - vel) / (DT * G), torch.zeros_like(vel))) * present[..., None]]
            upz += [((vel - pv) / (DT * G))[..., 2:3] * both[..., None],
                    torch.where(kin, (kn - vel) / (DT * G), torch.zeros_like(vel))[..., 2:3] * present[..., None]]
        feats += [grav, grip, ex, sl.reshape(N, Fw, -1), cf]
        if self.own_up:
            feats += [torch.cat(upz, -1) * cl[:, None, None]]
        body_tok = self._body(batch)
        h = (self.tok_in(torch.cat(feats, -1)) + body_tok[body_of][:, None]).reshape(N * Fw, -1)
        bounds = sample_bounds(batch)
        rs = self._rope_space(batch, Fw)
        rt = rope_angles(torch.arange(N * Fw, device=dev) % Fw, self.dh, ROPE_TIME)
        causal = torch.tril(torch.ones(Fw, Fw, dtype=torch.bool, device=dev))
        t_mask = ((causal[None] & ~batch["absent"][:, None, :]) | torch.eye(Fw, dtype=torch.bool, device=dev)[None])[:, None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            if self.bf16:
                h = h.bfloat16()
            for blk in self.blocks:
                if self.training and self.ckpt:
                    h = checkpoint(blk, h, Fw, bounds, rs, rt, t_mask, None, None, use_reentrant=False)
                else:
                    h = blk(h, Fw, bounds, rs, rt, t_mask, None, None)
        h = self.norm(h.float())
        w_tok = (present * given[None, :]).reshape(-1) * cl.repeat_interleave(Fw)
        pooled = torch.zeros(nb, self.d, device=dev).index_add_(0, body_of.repeat_interleave(Fw), h * w_tok[:, None])
        n_tok = torch.zeros(nb, device=dev).index_add_(0, body_of.repeat_interleave(Fw), w_tok)
        slot_in = torch.cat([pooled / n_tok.clamp_min(1)[:, None], body_tok], -1)
        if getattr(self, "slot_detach", False):
            slot_in = slot_in.detach()
        params_pred = self.param(slot_in)
        if self.prob_slots:
            self.slot_logvar = params_pred[:, self.n_params:].clamp(-10.0, 4.0)
            params_pred = params_pred[:, :self.n_params]
        if self.unc is not None:
            self.log_sigma = self.unc(h.detach()).view(N, Fw) - 7.0
        back = lambda f: torch.einsum("nfba,nfb->nfa", M, f)             # own frame -> world (M^T f)
        up = self.f_up(h).view(N, Fw, 2) if self.own_up else torch.zeros(N, Fw, 2, device=dev)
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev)
        f_int = (back(self.f_int(h).view(N, Fw, 3)) + up[..., 0:1] * ez) * cl[:, None, None]
        if self.direct:
            return f_int, torch.zeros_like(f_int), params_pred, ec
        has_c = torch.zeros(N * Fw, device=dev).index_add_(0, ec[0], torch.ones(ec.shape[1], device=dev)) > 0
        f_con = (back(self.f_con(h).view(N, Fw, 3)) + up[..., 1:2] * ez) * (has_c.view(N, Fw) & cl[:, None])[..., None]
        a = (batch["area"] * cl).repeat_interleave(Fw)                   # internal forces sum to zero over a cloth no
        tot = torch.zeros(nb * Fw, 3, device=dev).index_add_(0, key.reshape(-1), f_int.reshape(-1, 3) * a[:, None])   # one holds
        atot = torch.zeros(nb * Fw, device=dev).index_add_(0, key.reshape(-1), a).clamp_min(1e-12)
        mean = torch.where((cnt > 0)[:, None], torch.zeros_like(tot), tot / atot[:, None])
        f_int = (f_int.reshape(-1, 3) - mean[key.reshape(-1)]).view(N, Fw, 3) * cl[:, None, None]
        return f_int, f_con, params_pred, ec

    def forward(self, batch):
        pos, vel = batch["pos"], batch["vel"]
        N, Fw, _ = pos.shape
        dev = pos.device
        present = (~batch["absent"]).float()
        state = batch["f_state"]
        given = batch["frame_given"].float()
        f_given = torch.asinh(batch["f_ext"] * (state == 1)[..., None] / 0.01)
        log_area = self._log_measure(batch)
        if self.own:
            return self._forward_own(batch)
        feats = [vel * present[..., None], batch["rest"][:, None].expand(-1, Fw, -1),
                 Fn.one_hot(batch["role"], 3).float()[:, None].expand(-1, Fw, -1),
                 Fn.one_hot(batch["kind"], 3).float()[:, None].expand(-1, Fw, -1),
                 batch["held"].float()[..., None], f_given, Fn.one_hot(state.long(), 3).float(),
                 given[None, :, None].expand(N, -1, -1), present[..., None],
                 batch["tau"][None, :, None].expand(N, -1, -1),
                 batch["surface"][:, None].expand(-1, Fw, -1), batch["normal"][:, None].expand(-1, Fw, -1),
                 log_area[:, None, None].expand(-1, Fw, -1), batch["kin_next"]]
        if self.accel:                                               # (v_j - v_{j-1}) / dt, (v_{j+1} - v_j) / dt in g
            pv = torch.cat([vel[:, :1], vel[:, :-1]], 1)
            both = present * torch.cat([present[:, :1], present[:, :-1]], 1)
            kin = ((batch["kind"] == KIND["gripper"])[:, None] | batch["held"])[..., None]
            feats += [(vel - pv) / (DT * G) * both[..., None],
                      torch.where(kin, (batch["kin_next"] - vel) / (DT * G), torch.zeros_like(vel)) * present[..., None]]
        if self.oracle_v:                                            # the teacher's instantaneous velocity at the
            vt = batch["vtrue"][:, :Fw]                                  # clean past frames (record_v data): itself,
            on = batch.get("vtrue_on")                                   # its gap to the frame-average one in g, a
            if on is None:                                               # flag, the neighbours' relative to it
                on = (torch.arange(Fw, device=dev) < int(batch["n_past"]))[None].expand(N, -1)
            vm = (on & batch["is_cloth"][:, None]).float() * present
            feats += [vt * vm[..., None], (vt - vel) / (DT * G) * vm[..., None], vm[..., None],
                      slot_relvel(batch, vt) * vm[..., None]]
        M_loc = None
        if self.patch:
            st = batch.get("_stencil")
            if st is None or st[0].shape[0] != N:
                st = grid_stencil(batch, self.patch)
            pf, M_loc = patch_features(batch, pos, vel, *st, self.patch)
            feats += [pf * present[..., None]]
        ec, es = cloth_edges(batch, pos)
        feats += [self._slots(batch, pos, vel) * batch["is_cloth"][:, None, None],
                  contact_features(batch, pos, vel, ec, es) * present[..., None]]
        nb = int(batch["n_bodies"])
        body_of = batch["body_of"]
        body_tok = self._body(batch)
        h = (self.tok_in(torch.cat(feats, -1)) + body_tok[body_of][:, None]).reshape(N * Fw, -1)
        bounds = sample_bounds(batch)
        rs = self._rope_space(batch, Fw)
        rt = rope_angles(torch.arange(N * Fw, device=dev) % Fw, self.dh, ROPE_TIME)
        causal = torch.tril(torch.ones(Fw, Fw, dtype=torch.bool, device=dev))
        t_mask = (causal[None] & ~batch["absent"][:, None, :]) | torch.eye(Fw, dtype=torch.bool, device=dev)[None]
        t_mask = t_mask[:, None]                                         # (N, 1, Fw, Fw), True = may attend
        zemb = zidx = None
        if self.noise_dim:                                               # one Gaussian vector per body per frame (the
            z = batch.get("gen_z")                                       # global RNG in training: checkpoints replay it)
            if z is None:
                z = torch.randn(nb, Fw, self.noise_dim, device=dev, generator=self.gen_rng)
                if self.gen_zero:
                    z = torch.zeros_like(z)
            zemb = self.noise_in(z.reshape(nb * Fw, -1).float())
            zidx = (body_of[:, None] * Fw + torch.arange(Fw, device=dev)[None]).reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            if self.bf16:
                h = h.bfloat16()                                         # the residual stream in bf16, as Qwen
            for blk in self.blocks:
                if self.training and self.ckpt:
                    h = checkpoint(blk, h, Fw, bounds, rs, rt, t_mask, zemb, zidx, use_reentrant=False)
                else:
                    h = blk(h, Fw, bounds, rs, rt, t_mask, zemb, zidx)
        h = self.norm(h.float())
        cl_tok = batch["is_cloth"].repeat_interleave(Fw)
        w_tok = (present * given[None, :]).reshape(-1) * cl_tok
        pooled = torch.zeros(nb, self.d, device=dev).index_add_(0, body_of.repeat_interleave(Fw), h * w_tok[:, None])
        cnt = torch.zeros(nb, device=dev).index_add_(0, body_of.repeat_interleave(Fw), w_tok)
        slot_in = torch.cat([pooled / cnt.clamp_min(1)[:, None], body_tok], -1)
        if getattr(self, "slot_detach", False):                      # --side_apart: the slot loss trains its head only
            slot_in = slot_in.detach()
        params_pred = self.param(slot_in)
        if self.prob_slots:                                          # mean, log variance (clamped: 0.007-7 units)
            self.slot_logvar = params_pred[:, self.n_params:].clamp(-10.0, 4.0)
            params_pred = params_pred[:, :self.n_params]
        if self.unc is not None:                                     # log rms one-step error per (vertex, frame)
            self.log_sigma = self.unc(h.detach()).view(N, Fw) - 7.0
        f_int = self.f_int(h).view(N, Fw, 3)
        if self.patch:                                                   # M's rows are the axes: world = M^T local
            f_int = f_int + torch.einsum("nfba,nfb->nfa", M_loc, self.f_loc(h).view(N, Fw, 3))
        f_int = f_int * batch["is_cloth"][:, None, None]
        if self.direct:                                                  # --direct: the whole acceleration in g, no
            return f_int, torch.zeros_like(f_int), params_pred, ec       # exact layer (cloth_net.direct_step)
        has_c = torch.zeros(N * Fw, device=dev).index_add_(0, ec[0], torch.ones(ec.shape[1], device=dev)) > 0
        f_con = self.f_con(h).view(N, Fw, 3) * (has_c.view(N, Fw) & batch["is_cloth"][:, None])[..., None]
        # internal forces sum to zero over a cloth no one holds (as ClothSimulator)
        key = (body_of[:, None] * Fw + torch.arange(Fw, device=dev)[None]).reshape(-1)
        a = (batch["area"] * batch["is_cloth"]).repeat_interleave(Fw)
        tot = torch.zeros(nb * Fw, 3, device=dev).index_add_(0, key, f_int.reshape(-1, 3) * a[:, None])
        atot = torch.zeros(nb * Fw, device=dev).index_add_(0, key, a).clamp_min(1e-12)
        held_any = torch.zeros(nb * Fw, device=dev).index_add_(0, key, batch["held"].reshape(-1).float()) > 0
        mean = torch.where(held_any[:, None], torch.zeros_like(tot), tot / atot[:, None])
        f_int = (f_int.reshape(-1, 3) - mean[key]).view(N, Fw, 3) * batch["is_cloth"][:, None, None]
        return f_int, f_con, params_pred, ec


# ---------------------------------------------------------------------------------------------------------------------
# Beyond cloth: one network for cloth, rope and soft bodies (2026-10-30; DIRECTION.md "Beyond cloth")
# ---------------------------------------------------------------------------------------------------------------------
N_SLOT3 = 26                                       # 3D neighbour slots: the 26 directions of a cube's 3 x 3 x 3 cells
_DIRS3 = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)],
                      dtype=torch.float32)
_DIRS3 = _DIRS3 / _DIRS3.norm(dim=-1, keepdim=True)
MEASURE_REF = (1e-4, 1e-2, 1e-7)                   # m^2, m, m^3: a typical vertex's area / length / volume share


def neighbour_slots_3d(batch, pos, vel):
    """Each deformable vertex's mesh neighbours in 26 slots by the direction of their 3D rest offset -- the nearest of
    the 26 unit directions (+-1, 0) / norm -- as (N, Fw, 26 * SLOT_F). A cloth's in-plane offsets land in the 8
    in-plane slots (E, NE, N, ... as neighbour_slots), a rope's in the two along x; an irregular tet mesh can put
    several neighbours in one slot: their features are averaged (the last feature, present, then the share present)."""
    N, Fw, _ = pos.shape
    dev = pos.device
    e = batch["edges"]
    src, dst = torch.cat([e[:, 0], e[:, 1]]), torch.cat([e[:, 1], e[:, 0]])
    ro = (batch["rest"][dst] - batch["rest"][src]) * batch["size"][batch["sample_of"][src]][:, None]   # m
    slot = (ro @ _DIRS3.to(dev).T).argmax(-1)
    off = pos[dst] - pos[src]
    rl = ro.norm(dim=-1).clamp_min(1e-9)[:, None]
    sr = off.norm(dim=-1) / rl - 1.0
    present = (~batch["absent"][src] & ~batch["absent"][dst]).float()
    f = torch.cat([(ro / 0.01)[:, None].expand(-1, Fw, -1), off / 0.01, torch.asinh(1e2 * sr)[..., None],
                   torch.asinh(1e5 * sr)[..., None], vel[dst] - vel[src], torch.ones_like(sr)[..., None]], -1)
    f = f * present[..., None]
    key = src * N_SLOT3 + slot
    out = torch.zeros(N * N_SLOT3, Fw, SLOT_F, device=dev).index_add_(0, key, f)
    cnt = torch.zeros(N * N_SLOT3, device=dev).index_add_(0, key, torch.ones_like(key, dtype=torch.float32))
    out = out / cnt.clamp_min(1.0)[:, None, None]
    return out.view(N, N_SLOT3, Fw, SLOT_F).transpose(1, 2).reshape(N, Fw, N_SLOT3 * SLOT_F)


def rope_sections(dh):
    """The head dim split over the rest position's (x, y, z) for 3D rotary positions: three even sections, the first
    two equal (Qwen2-VL's M-RoPE splits its (t, h, w) unevenly the same way): 64 -> 22, 22, 20."""
    a = 2 * round(dh / 6)
    return a, a, dh - 2 * a


class ClothQwen3D(ClothQwen):
    """ClothQwen for every deformable material (`cloth_train --arch qwen3d`): one token per vertex of a cloth, a rope
    or a soft body, the same blocks, heads and exact layer. What changes from ClothQwen:

      rotary     space positions on the vertex's 3D rest position in cm (cloth (u, v, 0), rope (s, 0, 0), a soft body
                 its rest shape), the head dim in three sections (rope_sections: 22 / 22 / 20 of 64), each rotated by
                 one axis; a cloth's third axis is 0, so its section is not rotated (the identity)
      neighbours 26 slots by 3D rest direction (neighbour_slots_3d; a cloth fills the 8 in-plane ones)
      material   a learned embedding of the material (cloth / rope / soft, always given) added to the body token, so
                 to every token of its sample and to the slot head's input; the four slots are per material
                 (cloth_windows.SLOT_NAMES, normalised per material)
      measure    the log of each vertex's area / length / volume share over a typical one per material (MEASURE_REF)
    """
    N_SLOTS = N_SLOT3

    def __init__(self, d=256, layers=8, heads=4, kv_heads=2, mlp_hidden=768, **kw):
        super().__init__(d, layers, heads, kv_heads, mlp_hidden, **kw)
        assert min(rope_sections(self.dh)) > 0
        self.material = nn.Embedding(3, d)
        nn.init.normal_(self.material.weight, std=0.02)

    def _material(self, batch):
        m = batch.get("material")
        return torch.zeros(int(batch["n_bodies"]), dtype=torch.long, device=batch["pos"].device) if m is None else m

    def _log_measure(self, batch):
        ref = torch.as_tensor(MEASURE_REF, device=batch["area"].device)[self._material(batch)][batch["body_of"]]
        return torch.log(batch["area"].clamp_min(1e-12) / ref) * batch["is_cloth"]

    def _slots(self, batch, pos, vel):
        return neighbour_slots_3d(batch, pos, vel)

    def _body(self, batch):
        return super()._body(batch) + self.material(self._material(batch))

    def _rope_space(self, batch, Fw):
        p = batch["rest"] * batch["size"][batch["sample_of"]][:, None] / 0.01 * batch["is_cloth"][:, None]
        p = p.repeat_interleave(Fw, 0)                                   # per token, cm
        out = []
        for k, D in enumerate(rope_sections(self.dh)):
            out += rope_angles(p[:, k], D, ROPE_SPACE)
        return tuple(out)
