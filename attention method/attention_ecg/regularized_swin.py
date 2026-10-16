"""Isolated dropout adaptation; frozen v1 modules and parameter names stay intact."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import CepstralSwin, SwinBlock


class DropoutAttention(nn.Module):
    def __init__(self, original, probability):
        super().__init__()
        self.dim, self.heads = original.dim, original.heads
        self.window_size, self.shift = original.window_size, original.shift
        self.qkv, self.proj = original.qkv, original.proj
        self.relative_bias = original.relative_bias
        self.register_buffer('relative_index', original.relative_index.clone())
        self.attention_dropout = nn.Dropout(probability)
        self.projection_dropout = nn.Dropout(probability)

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
        attention = self.attention_dropout(scores.softmax(-1))
        out = (attention @ v).transpose(1, 2).reshape(b * nw, w, d)
        out = self.projection_dropout(self.proj(out)).reshape(b, nw * w, d)
        return out[:, left:left + t]


class DropoutMLP(nn.Sequential):
    def __init__(self, original, probability):
        super().__init__(*list(original.children()))
        self.dropout = nn.Dropout(probability)

    def forward(self, x):
        return self.dropout(self[2](self.dropout(self[1](self[0](x)))))


def make_model(config, dropout):
    if not 0 <= dropout < 1:
        raise ValueError('Invalid dropout probability')
    model = CepstralSwin(**config)
    for block in model.stages:
        if isinstance(block, SwinBlock):
            block.attn = DropoutAttention(block.attn, dropout)
            block.mlp = DropoutMLP(block.mlp, dropout)
    return model
