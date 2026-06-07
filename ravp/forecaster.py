#!/usr/bin/env python3
"""RA-VP trajectory forecaster: a Trajectron++-style CVAE operating on per-step
position deltas (GRU encoder + social pooling + latent z + GRU decoder).
``.sample(hist_d, neigh, K)`` draws K future trajectories (relative positions).
``load_forecaster(path)`` builds the model and loads a fine-tuned checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0


def deltas_from_pos(pos: torch.Tensor) -> torch.Tensor:
    """(B,T,2) relative positions (origin 0 at t=0) -> per-step deltas."""
    prev = torch.cat([torch.zeros_like(pos[:, :1]), pos[:, :-1]], dim=1)
    return pos - prev


def pos_from_deltas(deltas: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(deltas, dim=1)


def _gauss_nll(mu, logvar, target):
    logvar = logvar.clamp(LOGVAR_MIN, LOGVAR_MAX)
    return 0.5 * (logvar + (target - mu) ** 2 / logvar.exp()).sum(-1)


# ---------------------------------------------------------------------------
# Trajectron++-style CVAE
# ---------------------------------------------------------------------------
class TrajectronPP(nn.Module):
    def __init__(self, Th=20, Tf=20, H=128, Z=32):
        super().__init__()
        self.Tf, self.Z = Tf, Z
        self.enc_hist = nn.GRU(2, H, batch_first=True)
        self.enc_neigh = nn.Sequential(nn.Linear(4, H), nn.ReLU(), nn.Linear(H, H))
        self.enc_fut = nn.GRU(2, H, batch_first=True)
        self.prior = nn.Linear(2 * H, 2 * Z)
        self.poste = nn.Linear(3 * H, 2 * Z)
        self.dec_init = nn.Linear(2 * H + Z, H)
        self.dec = nn.GRUCell(2, H)
        self.out = nn.Linear(H, 4)  # mu(2), logvar(2)
        self.H = H

    def _context(self, hist_d, neigh):
        _, h = self.enc_hist(hist_d)
        h = h[-1]                                   # (B,H)
        s = self.enc_neigh(neigh).mean(dim=1)       # (B,H)
        return torch.cat([h, s], dim=-1)            # (B,2H)

    def _decode(self, c, z, sample=False):
        B = c.shape[0]
        hid = torch.tanh(self.dec_init(torch.cat([c, z], dim=-1)))
        inp = torch.zeros(B, 2, device=c.device)
        mus, deltas = [], []
        for _ in range(self.Tf):
            hid = self.dec(inp, hid)
            o = self.out(hid)
            mu, logvar = o[:, :2], o[:, 2:].clamp(LOGVAR_MIN, LOGVAR_MAX)
            if sample:
                d = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
            else:
                d = mu
            mus.append(torch.cat([mu, logvar], -1)); deltas.append(d)
            inp = d
        return torch.stack(deltas, 1), torch.stack(mus, 1)  # (B,Tf,2),(B,Tf,4)

    def nll_loss(self, hist_d, neigh, fut_d, beta=1.0):
        c = self._context(hist_d, neigh)
        _, hf = self.enc_fut(fut_d); hf = hf[-1]
        pr = self.prior(c); po = self.poste(torch.cat([c, hf], -1))
        mu_p, lv_p = pr[:, :self.Z], pr[:, self.Z:]
        mu_q, lv_q = po[:, :self.Z], po[:, self.Z:]
        z = mu_q + torch.randn_like(mu_q) * (0.5 * lv_q).exp()
        _, params = self._decode(c, z, sample=False)
        mu, logvar = params[..., :2], params[..., 2:]
        recon = _gauss_nll(mu, logvar, fut_d).sum(-1).mean()
        kl = 0.5 * (lv_p - lv_q + (lv_q.exp() + (mu_q - mu_p) ** 2) / lv_p.exp() - 1).sum(-1).mean()
        return recon + beta * kl

    @torch.no_grad()
    def sample(self, hist_d, neigh, K):
        c = self._context(hist_d, neigh)
        pr = self.prior(c); mu_p, lv_p = pr[:, :self.Z], pr[:, self.Z:]
        outs = []
        for _ in range(K):
            z = mu_p + torch.randn_like(mu_p) * (0.5 * lv_p).exp()
            deltas, _ = self._decode(c, z, sample=True)
            outs.append(pos_from_deltas(deltas))
        return torch.stack(outs, 1)  # (B,K,Tf,2)


def load_forecaster(ckpt_path, Th=20, Tf=20, device="cpu") -> "TrajectronPP":
    """Build a TrajectronPP and load a (fine-tuned) state-dict checkpoint."""
    dev = torch.device(device)
    model = TrajectronPP(Th=Th, Tf=Tf).to(dev)
    state = torch.load(str(Path(ckpt_path)), map_location=dev, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
