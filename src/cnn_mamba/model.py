"""CNN--bidirectional-Mamba model used by the Drosophila experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba3
from torch.utils.checkpoint import checkpoint


class BiMamba3Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, headdim: int):
        super().__init__()
        self.fw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)
        self.bw = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim)

    def forward(self, x):
        return self.fw(x) + self.bw(x.flip(1)).flip(1)


class CNNBiMamba3Block(nn.Module):
    def __init__(self, d_model, d_state, headdim, dropout, kernel_size=7):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("cnn_kernel_size must be a positive odd integer")
        self.cnn_norm = nn.LayerNorm(d_model)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.pointwise = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.cnn_drop = nn.Dropout(dropout)
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba = BiMamba3Block(d_model, d_state, headdim)
        self.mamba_drop = nn.Dropout(dropout)
        nn.init.ones_(self.cnn_norm.weight)
        nn.init.zeros_(self.cnn_norm.bias)
        nn.init.ones_(self.mamba_norm.weight)
        nn.init.zeros_(self.mamba_norm.bias)
        nn.init.trunc_normal_(self.depthwise.weight, std=0.02)
        nn.init.zeros_(self.depthwise.bias)
        nn.init.trunc_normal_(self.pointwise.weight, std=0.02)
        nn.init.zeros_(self.pointwise.bias)

    def forward(self, x):
        y = self.cnn_norm(x).transpose(1, 2)
        y = self.pointwise(F.gelu(self.depthwise(y))).transpose(1, 2)
        x = x + self.cnn_drop(y)
        return x + self.mamba_drop(self.mamba(self.mamba_norm(x)))


class SpliceMamba(nn.Module):
    """Two-head model: splice(background/donor/acceptor) and codon(background/start/stop)."""

    def __init__(self, d_model=192, d_state=64, n_layers=8, dropout=0.1,
                 architecture="cnn_mamba", cnn_kernel_size=7):
        super().__init__()
        if architecture not in {"mamba", "cnn_mamba"}:
            raise ValueError(f"unsupported architecture: {architecture}")
        self.architecture = architecture
        self.gradient_checkpointing = False
        self.embedding = nn.Embedding(5, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.embed_drop = nn.Dropout(dropout)
        if architecture == "mamba":
            self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
            self.layers = nn.ModuleList([
                BiMamba3Block(d_model, d_state, min(64, d_model))
                for _ in range(n_layers)
            ])
            self.drops = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([
                CNNBiMamba3Block(d_model, d_state, min(64, d_model), dropout, cnn_kernel_size)
                for _ in range(n_layers)
            ])
        self.final_norm = nn.LayerNorm(d_model)

        def head():
            return nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 3),
            )

        self.splice_head = head()
        self.start_stop_head = head()

    def forward(self, sequence):
        x = self.embed_drop(self.input_proj(self.embedding(sequence)))
        if self.architecture == "mamba":
            for norm, layer, drop in zip(self.norms, self.layers, self.drops):
                x = x + drop(layer(norm(x)))
        else:
            for layer in self.layers:
                if self.gradient_checkpointing and self.training:
                    x = checkpoint(layer, x, use_reentrant=True, preserve_rng_state=True)
                else:
                    x = layer(x)
        x = self.final_norm(x)
        return self.splice_head(x), self.start_stop_head(x)


def model_from_checkpoint(saved, device=None):
    config = saved.get("config", {})
    model = SpliceMamba(
        d_model=config.get("d_model", 192), d_state=config.get("d_state", 64),
        n_layers=config.get("n_layers", 8), dropout=config.get("dropout", 0.1),
        architecture=config.get("architecture", "cnn_mamba"),
        cnn_kernel_size=config.get("cnn_kernel_size", 7),
    )
    model.load_state_dict(saved["model_state_dict"])
    return model if device is None else model.to(device)
