"""AlphaQubit-style recurrent transformer decoder for surface codes.

A compact, faithful-in-spirit reference implementation of the neural decoder in

    J. Bausch et al., "Learning high-accuracy error decoding for quantum
    processors", Nature 636, 834-840 (2024).

The architecture keeps the three load-bearing ideas of the paper:

  * a per-stabilizer representation vector (one token per detector location);
  * a *syndrome transformer* that mixes stabilizer tokens each QEC round, with
    an attention bias derived from the 2D code geometry (the paper's
    spatial-structure prior);
  * a *recurrence* that carries a per-stabilizer state across rounds, so the
    decoder handles an arbitrary number of rounds at inference time.

A readout head pools the final state into a logical-error prediction.

Simplifications vs. the published model are documented in the project README
(no analog/soft inputs or leakage channel by default, learned pairwise spatial
bias in place of the dilated-convolution stack, GRU recurrence). The input
layer accepts an optional per-detector soft value, so analog inputs can be
added without changing the rest of the network.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AlphaQubitConfig:
    n_stab: int                 # stabilizers (detector locations) per round
    n_logical: int = 1          # logical observables to predict
    d_model: int = 128
    n_heads: int = 4
    n_blocks: int = 3           # syndrome-transformer blocks applied each round
    d_ff: int = 256
    dropout: float = 0.0
    soft_input: bool = False    # also consume a per-detector analog value
    use_spatial_bias: bool = True


class MultiHeadAttention(nn.Module):
    """Standard MHA with an optional additive (n_heads, N, N) bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.h, self.dk).transpose(1, 2)   # b,h,n,dk
        k = self.k(x).view(b, n, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(b, n, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)        # b,h,n,n
        if bias is not None:
            att = att + bias.unsqueeze(0)
        att = self.drop(att.softmax(dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(b, n, d)
        return self.o(out)


class SyndromeBlock(nn.Module):
    """Pre-norm transformer block mixing the stabilizer tokens."""

    def __init__(self, cfg: AlphaQubitConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), bias)
        x = x + self.ff(self.ln2(x))
        return x


class AlphaQubit(nn.Module):
    """Recurrent transformer surface-code decoder.

    Parameters
    ----------
    cfg : AlphaQubitConfig
    stab_coords : array-like of shape (n_stab, 2), optional
        2D coordinates of each stabilizer. Used to build the spatial attention
        bias (the code-geometry prior). If omitted, the bias is disabled.
    """

    def __init__(self, cfg: AlphaQubitConfig, stab_coords=None):
        super().__init__()
        self.cfg = cfg
        self.stab_embed = nn.Parameter(torch.randn(cfg.n_stab, cfg.d_model) * 0.02)
        in_dim = 2 if cfg.soft_input else 1
        self.in_proj = nn.Linear(in_dim, cfg.d_model)
        self.blocks = nn.ModuleList([SyndromeBlock(cfg) for _ in range(cfg.n_blocks)])
        self.gru = nn.GRUCell(cfg.d_model, cfg.d_model)
        self.read_ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.n_logical),
        )

        # Spatial attention bias: an MLP maps each (dx, dy) offset to a per-head
        # scalar, giving the attention a learnable, geometry-aware prior.
        self.bias_mlp = None
        self.register_buffer("rel", torch.zeros(0), persistent=False)
        if cfg.use_spatial_bias and stab_coords is not None:
            coords = torch.as_tensor(stab_coords, dtype=torch.float32)     # (N, 2)
            rel = coords[:, None, :] - coords[None, :, :]                  # (N, N, 2)
            # normalise offsets to keep the MLP input well-scaled
            scale = rel.abs().amax().clamp(min=1.0)
            self.register_buffer("rel", rel / scale, persistent=False)
            self.bias_mlp = nn.Sequential(
                nn.Linear(2, 32), nn.GELU(), nn.Linear(32, cfg.n_heads)
            )

    def _spatial_bias(self) -> torch.Tensor | None:
        if self.bias_mlp is None or self.rel.numel() == 0:
            return None
        b = self.bias_mlp(self.rel)             # (N, N, h)
        return b.permute(2, 0, 1).contiguous()  # (h, N, N)

    def forward(self, detectors: torch.Tensor, soft: torch.Tensor | None = None) -> torch.Tensor:
        """Predict logical-observable flips from a detection-event history.

        Parameters
        ----------
        detectors : (B, T, N) float tensor of detection events in {0, 1}
        soft : (B, T, N) float tensor of analog values, required iff
            cfg.soft_input is True.

        Returns
        -------
        logits : (B, n_logical) tensor (apply sigmoid for probabilities)
        """
        b, t, n = detectors.shape
        assert n == self.cfg.n_stab, f"expected {self.cfg.n_stab} stabilizers, got {n}"
        state = self.stab_embed.unsqueeze(0).expand(b, n, -1).contiguous()
        bias = self._spatial_bias()
        if bias is not None:
            bias = bias.to(detectors.dtype)

        for step in range(t):
            feat = detectors[:, step, :].unsqueeze(-1)                     # B,N,1
            if self.cfg.soft_input:
                assert soft is not None, "cfg.soft_input=True requires `soft`"
                feat = torch.cat([feat, soft[:, step, :].unsqueeze(-1)], dim=-1)
            h = state + self.in_proj(feat)                                 # inject round
            for blk in self.blocks:
                h = blk(h, bias)                                           # spatial mix
            state = self.gru(h.reshape(b * n, -1),
                             state.reshape(b * n, -1)).view(b, n, -1)      # recurrence

        pooled = self.read_ln(state).mean(dim=1)                           # B,D
        return self.head(pooled)                                           # B,n_logical

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
