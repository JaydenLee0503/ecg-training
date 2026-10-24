"""Unchanged temporal Swin with optional dropout; small raw-waveform control."""
from . import common as c
c.imports()
import torch
from torch import nn
from attention_ecg.regularized_swin import make_model


class ResidualBlock(nn.Module):
    def __init__(self, source, target):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(source,target,7,2,3,bias=False),
            nn.GroupNorm(4,target), nn.GELU(), nn.Conv1d(target,target,7,1,3,bias=False),
            nn.GroupNorm(4,target))
        self.skip = nn.Conv1d(source,target,1,2,bias=False)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.body(x)+self.skip(x))


class RawCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12,16,15,4,7,bias=False), nn.GroupNorm(4,16), nn.GELU())
        self.blocks = nn.Sequential(ResidualBlock(16,32), ResidualBlock(32,64), ResidualBlock(64,64))
        self.dropout = nn.Dropout(.1)
        self.head = nn.Linear(64,2)

    def forward(self, x):
        return self.head(self.dropout(self.blocks(self.stem(x)).mean(-1)))


def model(arm, spec):
    if arm == 'raw':
        return RawCNN()
    return make_model(spec['swin_model'], {'swin0':0., 'swin1':.1}[arm])
