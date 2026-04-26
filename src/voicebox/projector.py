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
    """Two routes from concept vector to voicebox modulation:

    1. **LoRA pairs** — per-layer (A, B) low-rank deltas added to attention
       output projections (the original OBJECTIVE.MD path).
    2. **Concept bias** — a single d_model vector added to every position's
       embedding. This is a wider, simpler routing channel that ensures the
       projector can inject information at position 0 (where the LoRA path
       has very little leverage).
    """

    def __init__(self, proj_cfg: ProjectorConfig, vb_cfg: VoiceboxConfig):
        super().__init__()
        self.n_layers = vb_cfg.n_layers
        self.d_model = vb_cfg.d_model
        self.rank = vb_cfg.lora_rank

        out_per_layer = 2 * self.d_model * self.rank  # one A and one B
        lora_total = self.n_layers * out_per_layer

        layers: list[nn.Module] = []
        in_dim = proj_cfg.teacher_hidden_dim
        for _ in range(proj_cfg.n_hidden_layers):
            layers += [
                nn.Linear(in_dim, proj_cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(proj_cfg.dropout),
            ]
            in_dim = proj_cfg.hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Two heads on top of the shared trunk.
        self.lora_head = nn.Linear(in_dim, lora_total)
        self.bias_head = nn.Linear(in_dim, self.d_model)

        # Zero-init both heads so the voicebox starts as its base weights —
        # standard LoRA convention; gives a clean optimization starting point.
        for head in (self.lora_head, self.bias_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, concept: torch.Tensor
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """concept: (B, teacher_hidden_dim).

        Returns:
            lora_pairs: list of (A, B) per layer
                A: (batch, d_model, rank)
                B: (batch, rank, d_model)
            concept_bias: (batch, d_model) — added to every position's embedding
        """
        batch = concept.size(0)
        h = self.trunk(concept)
        flat = self.lora_head(h).view(batch, self.n_layers, 2, self.d_model, self.rank)
        bias = self.bias_head(h)  # (B, d_model)

        pairs = []
        for i in range(self.n_layers):
            a = flat[:, i, 0]                      # (B, d_model, rank)
            b = flat[:, i, 1].transpose(-1, -2)    # (B, rank, d_model)
            pairs.append((a, b))
        return pairs, bias
