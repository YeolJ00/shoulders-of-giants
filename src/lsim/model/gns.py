"""Baseline B1 (DIRECTION.md "Baselines"): a GNS / MeshGraphNets-style next-frame model.

Encode-process-decode over the graph of each frame on its own (Sanchez-Gonzalez et al. 2020;
Pfaff et al. 2021): a vertex sees its last C velocities (no attention over a long history), its
body's given slots, its role and the given external force; edges are the contact pairs of that
frame (< 3 cm, different bodies) and, within a body, the pairs of its points closer than 2.5 cm
in the body frame (world-frame offsets, so the graph carries the body's pose); M message-passing
steps with unshared weights and residual updates; the decoder gives one acceleration per vertex.
Trained and rolled out exactly like B2 (`direct`: x' = 2x - x_prev + a dt^2, then the best rigid
placement per body), with the same noise and pushforward, so B1 vs B2 isolates the network and
B2 vs the model isolates the physics layer. It infers no slots: hidden parameters are zeros.
"""
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint

from lsim.model.net import R_CONTACT, contact_edges, mlp, rbf

R_BODY = 0.025


class GNS(nn.Module):
    def __init__(self, d=128, steps=10, n_params=11, C=5, ckpt=True):
        super().__init__()
        self.d, self.C, self.ckpt, self.n_params, self.direct = d, C, ckpt, n_params, True
        # C velocities, rest, given external force, role, force-slot state, given frame, tau,
        # present, surface material, body slots (value x given, given flags), body type
        self.node_in = mlp(3 * C + 3 + 3 + 3 + 3 + 1 + 1 + 1 + 2 + 2 * n_params + 4, d, d)
        self.edge_in = mlp(3 + 1 + 8 + 3 + 1, d, d)                   # offset, dist, rbf, rel vel, is-contact
        self.edge_mlp = nn.ModuleList([mlp(3 * d, d, d) for _ in range(steps)])
        self.node_mlp = nn.ModuleList([mlp(2 * d, d, d) for _ in range(steps)])
        self.ln_e = nn.ModuleList([nn.LayerNorm(d) for _ in range(steps)])
        self.ln_n = nn.ModuleList([nn.LayerNorm(d) for _ in range(steps)])
        self.acc = mlp(d, d, 3)
        nn.init.zeros_(self.acc[-1].weight); nn.init.zeros_(self.acc[-1].bias)   # start from constant velocity

    def _step(self, i, h, e, src, dst):
        e = e + self.edge_mlp[i](torch.cat([self.ln_e[i](e), h[src], h[dst]], -1))
        agg = torch.zeros(h.shape, device=h.device, dtype=torch.float32).index_add_(0, dst, e.float())
        return h + self.node_mlp[i](torch.cat([self.ln_n[i](h), agg.to(h.dtype)], -1)), e

    def forward(self, batch, slot_mass):
        """Accelerations (N, Fw, 3) for the frames from "now" (window frame P-1) on -- the only
        ones a step starts from; earlier frames are history and get zeros."""
        pos, vel = batch["pos"], batch["vel"]                          # (N, Fw, 3)
        N, Fw, _ = pos.shape
        dev = pos.device
        f0 = max(int(batch["n_past"]) - 1, 0)
        S = Fw - f0                                                    # frames computed
        present = (~batch["absent"]).float()
        hist = [torch.cat([torch.zeros_like(vel[:, :c]), vel[:, :Fw - c]], 1)[:, f0:] *
                torch.cat([torch.zeros_like(present[:, :c]), present[:, :Fw - c]], 1)[:, f0:, None] for c in range(self.C)]
        state = batch["f_state"][:, f0:]
        f_given = torch.asinh(batch["f_ext"][:, f0:] * (state == 1)[..., None] / 0.1)
        body = batch["body_of"]
        slot = torch.cat([batch["params"] * batch["param_given"], batch["param_given"].float(),
                          Fn.one_hot(batch["body_type"], 4).float()], -1)[body]
        feats = torch.cat(hist + [batch["rest"][:, None].expand(-1, S, -1), f_given,
                                  Fn.one_hot(batch["role"], 3).float()[:, None].expand(-1, S, -1),
                                  Fn.one_hot(state.long(), 3).float(),
                                  batch["frame_given"].float()[None, f0:, None].expand(N, -1, -1), present[:, f0:, None],
                                  batch["tau"][None, f0:, None].expand(N, -1, -1),
                                  batch["surface"][:, None].expand(-1, S, -1), slot[:, None].expand(-1, S, -1)], -1)
        h = self.node_in(feats).reshape(N * S, -1)
        # edges: contacts of each frame, and within-body neighbours (body frame; the same every frame)
        ps = pos[:, f0:].contiguous()
        ec = contact_edges(ps, body, batch["sample_of"], batch["role"] > 0)
        ib = contact_edges(batch["vert_local"][:, None], torch.arange(N, device=dev), body, batch["role"] > 0,
                           r=R_BODY)          # token = vertex (one frame); static points (local coords 0) get none
        ib = (ib[:, :, None] * S + torch.arange(S, device=dev)).reshape(2, -1)
        gone = batch["absent"][:, f0:].reshape(-1)
        x, vt = ps.reshape(-1, 3), vel[:, f0:].reshape(-1, 3)
        es, fs = [], []
        for e, is_c in ((ib, 0.0), (ec, 1.0)):
            e = e[:, ~(gone[e[0]] | gone[e[1]])]
            off = x[e[0]] - x[e[1]]
            dist = torch.linalg.norm(off, dim=-1)
            fs.append(torch.cat([off / R_CONTACT, dist[:, None] / R_CONTACT, rbf(dist), vt[e[0]] - vt[e[1]],
                                 torch.full_like(dist[:, None], is_c)], -1))
            es.append(e)
        src, dst = torch.cat(es, 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = self.edge_in(torch.cat(fs, 0))
            for i in range(len(self.edge_mlp)):
                if self.training and self.ckpt:
                    h, e = checkpoint(self._step, i, h, e, src, dst, use_reentrant=False)
                else:
                    h, e = self._step(i, h, e, src, dst)
        acc = torch.zeros(N, Fw, 3, device=dev)
        acc[:, f0:] = 9.81 * self.acc(h.float()).view(N, S, 3)
        nb = int(batch["n_bodies"])
        return acc, torch.zeros_like(acc), torch.zeros(nb, self.n_params, device=dev)
