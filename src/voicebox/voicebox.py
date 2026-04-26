"""Voicebox: the tiny micro-decoder whose weights are JIT-modulated by LoRA
deltas coming from the HyperProjector.

This is intentionally minimal — a standard causal decoder transformer with one
LoRA-modulated weight per block (the attention output projection). Everything
else is static. Status: scaffold; the training loop is not yet wired in.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VoiceboxConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: VoiceboxConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.d_model = cfg.d_model

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        # Output projection — base weight is static, modulated at runtime.
        self.out_base = nn.Parameter(torch.empty(cfg.d_model, cfg.d_model))
        nn.init.xavier_uniform_(self.out_base)

    def forward(self, x: torch.Tensor, dyn_delta: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D); dyn_delta: (B, D, D) added to out projection."""
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, d_head)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, T, d_head)

        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).contiguous().view(B, T, D)

        # Per-sample modulated output projection: y = att @ (W_base + delta)^T
        w = self.out_base.unsqueeze(0) + dyn_delta  # (B, D, D)
        return torch.einsum("btd,bod->bto", att, w)


class FeedForward(nn.Module):
    def __init__(self, cfg: VoiceboxConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, cfg: VoiceboxConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x: torch.Tensor, dyn_delta: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), dyn_delta)
        x = x + self.ff(self.ln2(x))
        return x


class Voicebox(nn.Module):
    def __init__(self, cfg: VoiceboxConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie embeddings <-> output projection for parameter efficiency.
        self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        tokens: torch.Tensor,
        lora_pairs: list[tuple[torch.Tensor, torch.Tensor]],
        concept_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """tokens: (B, T); lora_pairs: list of (A, B) per layer;
        concept_bias: (B, D) added to every position's embedding (optional).

        Returns logits (B, T, vocab_size).
        """
        B, T = tokens.shape
        assert T <= self.cfg.max_seq_len
        assert len(lora_pairs) == self.cfg.n_layers

        pos = torch.arange(T, device=tokens.device).unsqueeze(0)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        if concept_bias is not None:
            x = x + concept_bias.unsqueeze(1)
        for blk, (a, b) in zip(self.blocks, lora_pairs):
            dyn_delta = torch.bmm(a, b) / math.sqrt(self.cfg.lora_rank)
            x = blk(x, dyn_delta)
        x = self.ln_f(x)
        return self.lm_head(x)
