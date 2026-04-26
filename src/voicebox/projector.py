"""Hyper-Projector: maps a teacher concept vector to LoRA (A, B) weight pairs.

For each modulated voicebox layer i, we emit two low-rank matrices A_i, B_i
such that the dynamic weight delta is A_i @ B_i (shape d_model x d_model).
The projector total output is small (n_layers * 2 * d_model * rank), so the
whole thing remains tractable even with much bigger voiceboxes.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ProjectorConfig, VoiceboxConfig


class HyperProjector(nn.Module):
    def __init__(self, proj_cfg: ProjectorConfig, vb_cfg: VoiceboxConfig):
        super().__init__()
        self.n_layers = vb_cfg.n_layers
        self.d_model = vb_cfg.d_model
        self.rank = vb_cfg.lora_rank

        out_per_layer = 2 * self.d_model * self.rank  # one A and one B
        total_out = self.n_layers * out_per_layer

        layers: list[nn.Module] = []
        in_dim = proj_cfg.teacher_hidden_dim
        for _ in range(proj_cfg.n_hidden_layers):
            layers += [nn.Linear(in_dim, proj_cfg.hidden_dim), nn.GELU()]
            in_dim = proj_cfg.hidden_dim
        layers.append(nn.Linear(in_dim, total_out))
        self.trunk = nn.Sequential(*layers)

        # Zero-init the final layer so that at step 0 the dynamic delta is 0
        # and the voicebox behaves like its base weights — a clean starting
        # point for training (LoRA convention).
        nn.init.zeros_(self.trunk[-1].weight)
        nn.init.zeros_(self.trunk[-1].bias)

    def forward(self, concept: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """concept: (B, teacher_hidden_dim) -> list of (A, B) per layer.

        A: (batch, d_model, rank)
        B: (batch, rank, d_model)
        """
        batch = concept.size(0)
        flat = self.trunk(concept)  # (B, n_layers * 2 * d_model * rank)
        flat = flat.view(batch, self.n_layers, 2, self.d_model, self.rank)
        out = []
        for i in range(self.n_layers):
            a = flat[:, i, 0]                      # (B, d_model, rank)
            b = flat[:, i, 1].transpose(-1, -2)    # (B, rank, d_model)
            out.append((a, b))
        return out
