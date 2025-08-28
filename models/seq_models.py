from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUClassifier(nn.Module):
    """
    Small GRU baseline for per-timestep binary classification on sequences.

    Inputs:
      - x:  (B, T, F) float features
      - mask: (B, T) bool/int mask where 1 means valid timestep
      - course_ids: Optional (B,) long indices for per-course embedding (broadcast to timesteps)

    Outputs:
      - logits: (B, T) raw logits per timestep
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        course_vocab: Optional[int] = None,
        course_emb_dim: int = 16,
    ):
        super().__init__()
        self.course_emb = None
        if course_vocab is not None and course_vocab > 0 and course_emb_dim > 0:
            self.course_emb = nn.Embedding(course_vocab, course_emb_dim)
            nn.init.normal_(self.course_emb.weight, std=0.02)
            input_dim = input_dim + course_emb_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        course_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, F)
        if self.course_emb is not None and course_ids is not None:
            ce = self.course_emb(course_ids)  # (B, E)
            ce = ce.unsqueeze(1).expand(-1, x.size(1), -1)  # (B, T, E)
            x = torch.cat([x, ce], dim=-1)

        out, _ = self.gru(x)  # (B, T, H)
        logits = self.head(out).squeeze(-1)  # (B, T)
        if mask is not None:
            logits = logits * mask
        return logits


class LSTMClassifier(nn.Module):
    """
    Small LSTM baseline for per-timestep binary classification on sequences.

    Inputs:
      - x:  (B, T, F) float features
      - mask: (B, T) mask where 1 means valid timestep
      - course_ids: Optional (B,) long indices for per-course embedding

    Outputs:
      - logits: (B, T)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        course_vocab: Optional[int] = None,
        course_emb_dim: int = 16,
    ):
        super().__init__()
        self.course_emb = None
        if course_vocab is not None and course_vocab > 0 and course_emb_dim > 0:
            self.course_emb = nn.Embedding(course_vocab, course_emb_dim)
            nn.init.normal_(self.course_emb.weight, std=0.02)
            input_dim = input_dim + course_emb_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        course_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.course_emb is not None and course_ids is not None:
            ce = self.course_emb(course_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, ce], dim=-1)
        out, _ = self.lstm(x)
        logits = self.head(out).squeeze(-1)
        if mask is not None:
            logits = logits * mask
        return logits


class TimePositionalEncoding(nn.Module):
    """Relative/continuous time encoding using log-delta and Fourier features."""

    def __init__(self, d_model: int, n_freqs: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_freqs = n_freqs
        # Learnable projection from [log_dt, abs_pos] -> d_model
        self.proj = nn.Sequential(
            nn.Linear(2 + 2 * n_freqs, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, dt: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # dt, pos: (B, T)
        log_dt = torch.log1p(dt.clamp_min(0))
        # Fourier features on log_dt
        freqs = torch.arange(1, self.n_freqs + 1, device=dt.device, dtype=dt.dtype)
        freqs = freqs.view(1, 1, -1)  # (1,1,F)
        phase = log_dt.unsqueeze(-1) * freqs  # (B,T,F)
        sin = torch.sin(phase)
        cos = torch.cos(phase)
        feats = torch.cat([log_dt.unsqueeze(-1), pos.unsqueeze(-1), sin, cos], dim=-1)  # (B,T, 2+2F)
        return self.proj(feats)  # (B,T,d_model)


class TimeAwareTransformer(nn.Module):
    """
    Time-aware Transformer encoder for per-timestep binary classification.

    Inputs:
      - x: (B, T, F) features (normalized counts etc.)
      - dt: (B, T) inter-event time deltas (seconds or normalized)
      - mask: (B, T) bool mask where 1=valid
      - course_ids: Optional (B,) long indices for per-course embedding
    Outputs:
      - logits: (B, T)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        course_vocab: Optional[int] = None,
        course_emb_dim: int = 16,
        time_freqs: int = 8,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.time_enc = TimePositionalEncoding(d_model=d_model, n_freqs=time_freqs)
        self.course_emb = None
        if course_vocab is not None and course_vocab > 0 and course_emb_dim > 0:
            self.course_emb = nn.Embedding(course_vocab, d_model)
            nn.init.normal_(self.course_emb.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        course_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B,T,F); dt: (B,T)
        h = self.input_proj(x)  # (B,T,D)
        # absolute position index (0..T-1)
        pos = torch.arange(h.size(1), device=h.device, dtype=h.dtype).view(1, -1).expand(h.size(0), -1)
        tenc = self.time_enc(dt=dt, pos=pos)  # (B,T,D)
        h = h + tenc
        if self.course_emb is not None and course_ids is not None:
            ce = self.course_emb(course_ids)  # (B,D)
            h = h + ce.unsqueeze(1)

        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)  # True where padding
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        logits = self.head(h).squeeze(-1)  # (B,T)
        if mask is not None:
            logits = logits * mask
        return logits


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    pos_weight: Optional[float] = None,
) -> torch.Tensor:
    """Masked BCE with optional positive class weighting.

    Args:
        logits: (B,T)
        targets: (B,T) in {0,1}
        mask: (B,T) 1 for valid positions
        pos_weight: if provided, scales loss for positive targets
    """
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if pos_weight is not None and pos_weight > 0:
        w = torch.ones_like(loss)
        w = torch.where(targets >= 0.5, w * float(pos_weight), w)
        loss = loss * w
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom


class CausalConv1d(nn.Module):
    """1D causal convolution (pads on the left only)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))
        x = self.conv(x)
        x = self.dropout(x)
        return x


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation, dropout=dropout)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation, dropout=dropout)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        return x + h


class TCNClassifier(nn.Module):
    """
    Temporal Convolutional Network baseline for per-timestep binary classification.

    Inputs:
      - x: (B, T, F) features (includes log_dt as a feature)
      - mask: (B, T) optional mask
      - course_ids: Optional (B,) for course embeddings (concatenated to features)

    Output: logits (B, T)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        course_vocab: Optional[int] = None,
        course_emb_dim: int = 16,
    ):
        super().__init__()
        self.course_emb = None
        in_dim = input_dim
        if course_vocab is not None and course_vocab > 0 and course_emb_dim > 0:
            self.course_emb = nn.Embedding(course_vocab, course_emb_dim)
            nn.init.normal_(self.course_emb.weight, std=0.02)
            in_dim += course_emb_dim

        self.input_proj = nn.Linear(in_dim, d_model)
        blocks = []
        for i in range(num_layers):
            dilation = 2 ** i
            blocks.append(TCNBlock(d_model, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        course_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (B, T, F)
        if self.course_emb is not None and course_ids is not None:
            ce = self.course_emb(course_ids).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, ce], dim=-1)
        h = self.input_proj(x)  # (B,T,D)
        h = h.transpose(1, 2)  # (B,D,T)
        h = self.net(h)  # (B,D,T)
        h = h.transpose(1, 2)  # (B,T,D)
        logits = self.head(h).squeeze(-1)
        if mask is not None:
            logits = logits * mask
        return logits
    else:
        return loss.mean()
