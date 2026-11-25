"""Sampling a chunk for models trained with step_loss_chunk (the ablation; models trained with
the default step loss roll out frame by frame, lsim.model.rollout). The chunk starts as noise
(tau = 1) and is denoised in
`steps` deterministic steps. At each step the network sees the noisy chunk, predicts forces and
slots, the physics layer turns them into a clean estimate, and the noisy chunk is moved toward
it (DDIM on the per-frame displacements of dynamic vertices; the gripper's path, static and
hinged parts stay as given).
"""
import torch

from lsim.train.train import DT, SIGMA, integrate, slots
from lsim.train.windows import F_EMPTY, F_GIVEN


@torch.no_grad()
def sample_chunk(model, batch, K, steps=10, generator=None):
    """Returns the clean chunk positions (N, K, 3) after `steps` denoising steps, the first step's
    estimate (a single prediction from the past alone), and the final parameter predictions."""
    P = batch["n_past"]
    pos = batch["pos"]
    dev = pos.device
    dyn = (batch["role"] == 2) & (batch["type"] == 0)
    taus = torch.linspace(1.0, 0.0, steps + 1, device=dev)
    eps = torch.randn(pos.shape[0], K, 3, device=dev, generator=generator)
    d_tau = eps * SIGMA
    state = batch["f_state"]
    first = None
    for i in range(steps):
        tau, tau_next = taus[i], taus[i + 1]
        pos_in = pos.clone()
        pos_in[dyn, P:P + K] = (pos[:, P - 1:P] + torch.cumsum(d_tau, 1))[dyn]
        prev = torch.cat([pos_in[:, :1] - batch["vel"][:, :1] * DT, pos_in[:, :-1]], 1)
        b_in = dict(batch, pos=pos_in, vel=(pos_in - prev) / DT,
                    tau=torch.cat([torch.zeros(P, device=dev), tau.expand(K)]),
                    frame_given=torch.cat([torch.ones(P, device=dev), torch.zeros(K, device=dev)]))
        f_contact, f_ext_pred, pred = model(b_in, lambda pr: slots(batch, pr)[0])
        m, c, J = slots(batch, pred)
        f_ext = torch.where((state == F_GIVEN)[..., None], batch["f_ext"], torch.zeros_like(batch["f_ext"]))
        f_ext = torch.where((state == F_EMPTY)[..., None], f_ext_pred, f_ext)
        x0 = integrate(batch, f_contact + f_ext, K, P, m, c, J)
        if first is None:
            first = x0
        d0 = torch.diff(torch.cat([pos[:, P - 1:P], x0], 1), dim=1)
        if tau_next > 0:
            eps_hat = (d_tau - (1 - tau) * d0) / (tau * SIGMA)
            d_tau = (1 - tau_next) * d0 + tau_next * SIGMA * eps_hat
    return x0, first, pred
