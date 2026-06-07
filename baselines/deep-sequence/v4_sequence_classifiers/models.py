"""Sequence-classifier model family: LSTM-CLS, GRU-CLS, TCN-CLS.

All three models share the same input/output signature:

    forward(dyn: (B, T, D_dyn), static: (B, D_static), mask: (B, T)) -> (B,) logits

Only the temporal encoder differs. The classification head is identical across
models.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared classification head
# ---------------------------------------------------------------------------


class SeqClassifierHead(nn.Module):
    """Static-feature MLP + [h_seq, h_static] classification MLP.

    Head architecture (from the plan):
        Linear -> 128 -> ReLU -> Dropout
        Linear -> 64  -> ReLU -> Dropout
        Linear -> 1   (raw logit)
    """

    def __init__(self, h_seq_dim: int, d_static: int,
                 static_hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.static_mlp = nn.Sequential(
            nn.Linear(d_static, static_hidden),
            nn.ReLU(),
            nn.Linear(static_hidden, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        in_dim = h_seq_dim + static_hidden
        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, h_seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        h_st = self.static_mlp(static)
        h = torch.cat([h_seq, h_st], dim=-1)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: (B, T, C), mask: (B, T). Return (B, C) — mean over valid steps."""
    m = mask.unsqueeze(-1).float()
    s = (x * m).sum(dim=1)
    d = m.sum(dim=1).clamp(min=1.0)
    return s / d


class LSTMClassifier(nn.Module):
    """2-layer unidirectional LSTM. Take top-layer final hidden state by
    default; `pool='mean'` switches to masked temporal mean pool."""

    def __init__(self, d_in: int, d_static: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, pool: str = "last"):
        super().__init__()
        self.pool = pool
        self.rnn = nn.LSTM(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = SeqClassifierHead(
            h_seq_dim=hidden_size, d_static=d_static, dropout=dropout,
        )

    def forward(self, dyn: torch.Tensor, static: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        out, (h_n, _c_n) = self.rnn(dyn)
        if self.pool == "mean":
            h_seq = _masked_mean(out, mask)
        else:
            h_seq = h_n[-1]
        h_seq = self.dropout(h_seq)
        return self.head(h_seq, static)


class GRUClassifier(nn.Module):
    def __init__(self, d_in: int, d_static: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout: float = 0.2, pool: str = "last"):
        super().__init__()
        self.pool = pool
        self.rnn = nn.GRU(
            input_size=d_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = SeqClassifierHead(
            h_seq_dim=hidden_size, d_static=d_static, dropout=dropout,
        )

    def forward(self, dyn: torch.Tensor, static: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        out, h_n = self.rnn(dyn)
        if self.pool == "mean":
            h_seq = _masked_mean(out, mask)
        else:
            h_seq = h_n[-1]
        h_seq = self.dropout(h_seq)
        return self.head(h_seq, static)


# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------


class _CausalConv1d(nn.Module):
    """1D causal convolution with dilation — left-pad (k-1)*d then trim."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    """Two causal convs with residual."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.drop(h)
        h = self.act(self.conv2(h))
        h = self.drop(h)
        return h + self.res(x)


class TCNClassifier(nn.Module):
    """Causal TCN over the time axis. pool='mean' -> global avg over valid
    steps; pool='last' -> take the last-step output."""

    def __init__(self, d_in: int, d_static: int,
                 channels: List[int] = (64, 64, 128, 128),
                 kernel_size: int = 3, dilations: List[int] = (1, 2, 4, 8),
                 dropout: float = 0.1, pool: str = "mean"):
        super().__init__()
        assert len(channels) == len(dilations), "channels/dilations length mismatch"
        self.pool = pool
        blocks: List[nn.Module] = []
        prev = d_in
        for c, d in zip(channels, dilations):
            blocks.append(_TCNBlock(prev, c, kernel_size, d, dropout))
            prev = c
        self.blocks = nn.Sequential(*blocks)
        self.last_ch = prev
        self.head = SeqClassifierHead(
            h_seq_dim=prev, d_static=d_static, dropout=max(dropout, 0.2),
        )

    def forward(self, dyn: torch.Tensor, static: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # (B, T, D) -> (B, D, T)
        x = dyn.transpose(1, 2)
        h = self.blocks(x)  # (B, C, T)
        if self.pool == "last":
            h_seq = h[..., -1]
        else:
            h_seq = _masked_mean(h.transpose(1, 2), mask)
        return self.head(h_seq, static)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


MODEL_NAMES = ("lstm", "gru", "tcn")


def build_model(model_name: str, d_in: int, d_static: int,
                config: Dict) -> nn.Module:
    m = model_name.lower()
    if m == "lstm":
        c = config.get("lstm", {})
        return LSTMClassifier(
            d_in=d_in, d_static=d_static,
            hidden_size=int(c.get("hidden_size", 128)),
            num_layers=int(c.get("num_layers", 2)),
            dropout=float(c.get("dropout", 0.2)),
            pool=str(c.get("pool", "last")),
        )
    if m == "gru":
        c = config.get("gru", {})
        return GRUClassifier(
            d_in=d_in, d_static=d_static,
            hidden_size=int(c.get("hidden_size", 128)),
            num_layers=int(c.get("num_layers", 2)),
            dropout=float(c.get("dropout", 0.2)),
            pool=str(c.get("pool", "last")),
        )
    if m == "tcn":
        c = config.get("tcn", {})
        return TCNClassifier(
            d_in=d_in, d_static=d_static,
            channels=list(c.get("channels", [64, 64, 128, 128])),
            kernel_size=int(c.get("kernel_size", 3)),
            dilations=list(c.get("dilations", [1, 2, 4, 8])),
            dropout=float(c.get("dropout", 0.1)),
            pool=str(c.get("pool", "mean")),
        )
    raise ValueError(f"unknown model {model_name!r} (choices: {MODEL_NAMES})")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
