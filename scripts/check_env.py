"""Sanity check: verify torch/CUDA/MPS, instantiate the voicebox stack, run one
forward pass with a dummy concept vector and dummy tokens.

Usage:
    python scripts/check_env.py
"""
from __future__ import annotations

import torch

from voicebox.config import Config
from voicebox.projector import HyperProjector
from voicebox.voicebox import Voicebox


def main() -> None:
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"mps available: {torch.backends.mps.is_available()}")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"using device: {device}")

    cfg = Config()
    projector = HyperProjector(cfg.projector, cfg.voicebox).to(device)
    voicebox = Voicebox(cfg.voicebox).to(device)

    p_params = sum(p.numel() for p in projector.parameters())
    v_params = sum(p.numel() for p in voicebox.parameters())
    print(f"projector params: {p_params:,}")
    print(f"voicebox params:  {v_params:,}")

    B, T = 2, 16
    concept = torch.randn(B, cfg.projector.teacher_hidden_dim, device=device)
    tokens = torch.randint(0, cfg.voicebox.vocab_size, (B, T), device=device)

    pairs, bias = projector(concept)
    logits = voicebox(tokens, pairs, concept_bias=bias)
    print(f"logits shape: {tuple(logits.shape)}")
    assert logits.shape == (B, T, cfg.voicebox.vocab_size)
    print("OK")


if __name__ == "__main__":
    main()
