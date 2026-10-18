"""Swin-style 1-D hierarchy over ECG cepstral frames, randomly initialized.

Shifted windows use explicit left padding rather than cyclic rolling. Masked
padding prevents both edge wraparound and attention to nonexistent frames.
This is a temporal adaptation, not the original 2-D image Swin architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window_size=8, shift=0):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or not 0 <= shift < window_size:
            raise ValueError('Invalid attention dimensions or shift')
        self.dim, self.heads, self.window_size, self.shift = dim, heads, window_size, shift
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.relative_bias = nn.Parameter(torch.zeros(2 * window_size - 1, heads))
        pos = torch.arange(window_size)
        self.register_buffer('relative_index', pos[:, None] - pos[None, :] + window_size - 1)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x):
        b, t, d = x.shape
        if t < 1 or d != self.dim:
            raise ValueError('Invalid attention input')
        w = self.window_size
        left = self.shift if t > w else 0
        right = (-(t + left)) % w
        padded = F.pad(x, (0, 0, left, right))
        valid = F.pad(torch.ones(t, dtype=torch.bool, device=x.device), (left, right))
        nw = padded.shape[1] // w
        chunks = padded.reshape(b * nw, w, d)
        qkv = self.qkv(chunks).reshape(b * nw, w, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * ((d // self.heads) ** -0.5)) @ k.transpose(-2, -1)
        bias = self.relative_bias[self.relative_index].permute(2, 0, 1)
        scores = scores + bias[None]
        valid_keys = valid.reshape(nw, w).repeat(b, 1)
        scores = scores.masked_fill(~valid_keys[:, None, None, :], float('-inf'))
        out = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b * nw, w, d)
        out = self.proj(out).reshape(b, nw * w, d)
        return out[:, left:left + t]


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, window_size, shift):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window_size, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x):
        if x.shape[1] % 2:
            x = F.pad(x, (0, 0, 0, 1))
        return self.reduction(self.norm(torch.cat((x[:, 0::2], x[:, 1::2]), dim=-1)))


class CepstralSwin(nn.Module):
    """(batch, time, leads, cepstra) -> uncalibrated class logits.

    Each frame is one patch; flatten lead/cepstral axes into channels. Preserve
    the temporal axis for local/shifted attention and pairwise patch merging.
    Inputs in one batch must have the same real frame count (no batch padding).
    """
    def __init__(self, n_leads=12, n_ceps=20, num_classes=2, embed_dim=64,
                 depths=(2, 2, 2), heads=(2, 4, 8), window_size=8):
        super().__init__()
        if (not depths or len(depths) != len(heads) or any(d < 2 for d in depths)
                or min(n_leads, n_ceps, embed_dim) < 1 or num_classes < 2 or window_size < 2):
            raise ValueError('Invalid model configuration')
        self.n_leads, self.n_ceps = n_leads, n_ceps
        self.config = dict(n_leads=n_leads, n_ceps=n_ceps, num_classes=num_classes,
                           embed_dim=embed_dim, depths=list(depths), heads=list(heads),
                           window_size=window_size)
        self.embedding = nn.Sequential(nn.Linear(n_leads * n_ceps, embed_dim), nn.LayerNorm(embed_dim))
        layers = []
        for stage, depth in enumerate(depths):
            dim = embed_dim * 2 ** stage
            for block in range(depth):
                layers.append(SwinBlock(dim, heads[stage], window_size,
                                        0 if block % 2 == 0 else window_size // 2))
            if stage < len(depths) - 1:
                layers.append(PatchMerging(dim))
        self.stages = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        if (x.ndim != 4 or x.shape[1] < 1 or tuple(x.shape[2:]) != (self.n_leads, self.n_ceps)
                or not torch.isfinite(x).all()):
            raise ValueError('Expected finite (batch, time, configured leads, configured cepstra)')
        x = self.embedding(x.flatten(2))
        return self.head(self.norm(self.stages(x)).mean(dim=1))
