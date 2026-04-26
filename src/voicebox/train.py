"""Training-loop building blocks.

Trainable params: projector + voicebox. Teacher is frozen and not even loaded
here — its hidden states already live in the saved shard.

Loss: masked cross-entropy on target tokens, with a pad-token shifted in at
position 0 so the voicebox has to predict target[0] from the LoRA modulation
alone (rather than from target[0] itself).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config, ProjectorConfig, TrainConfig, VoiceboxConfig
from .data import VectorDataset, VectorShard
from .projector import HyperProjector
from .voicebox import Voicebox


# --- Optimizer / schedule ----------------------------------------------------

def build_optimizer(
    projector: HyperProjector, voicebox: Voicebox, cfg: TrainConfig
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for module in (projector, voicebox):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            # No weight decay on biases, layernorms, embeddings.
            if name.endswith(".bias") or "ln" in name or "emb" in name:
                no_decay.append(p)
            else:
                decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.n_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


# --- Forward + loss ----------------------------------------------------------

def shift_for_teacher_forcing(
    target_ids: torch.Tensor, pad_id: int
) -> torch.Tensor:
    """Prepend pad_id at position 0 so position t predicts target_ids[t]."""
    B, T = target_ids.shape
    pad_col = torch.full((B, 1), pad_id, dtype=target_ids.dtype, device=target_ids.device)
    return torch.cat([pad_col, target_ids[:, :-1]], dim=1)


def compute_loss(
    projector: HyperProjector,
    voicebox: Voicebox,
    concept: torch.Tensor,         # (B, teacher_hidden_dim)
    target_ids: torch.Tensor,      # (B, T)
    target_mask: torch.Tensor,     # (B, T) bool
    pad_id: int,
) -> tuple[torch.Tensor, dict]:
    pairs = projector(concept)
    inputs = shift_for_teacher_forcing(target_ids, pad_id)
    logits = voicebox(inputs, pairs)  # (B, T, V)

    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = target_ids.reshape(-1)
    flat_mask = target_mask.reshape(-1)

    losses = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    n_tokens = flat_mask.sum().clamp(min=1)
    loss = (losses * flat_mask.float()).sum() / n_tokens

    with torch.no_grad():
        preds = flat_logits.argmax(dim=-1)
        correct = ((preds == flat_labels) & flat_mask).float().sum()
        token_acc = (correct / n_tokens).item()
        # First-token accuracy: position 0 of every sample (most important —
        # this is the position with no prior target token, only LoRA signal).
        first_preds = logits[:, 0].argmax(dim=-1)
        first_correct = ((first_preds == target_ids[:, 0]) & target_mask[:, 0]).float()
        first_acc = first_correct.sum().item() / max(1, target_mask[:, 0].sum().item())

    return loss, {"loss": loss.item(), "token_acc": token_acc, "first_acc": first_acc}


# --- Train state -------------------------------------------------------------

@dataclass
class TrainState:
    projector: HyperProjector
    voicebox: Voicebox
    optimizer: torch.optim.Optimizer
    pad_id: int
    cfg: Config
    device: torch.device
    step: int = 0
    best_eval_loss: float = float("inf")


def make_state_from_shard(
    shard: VectorShard, cfg: Config, device: str | torch.device
) -> TrainState:
    """Override config dims from the shard, then build modules + optimizer."""
    cfg.projector.teacher_hidden_dim = shard.teacher_hidden_dim
    # Qwen's tokenizer.vocab_size excludes added special tokens, so token IDs
    # in target_ids can exceed it (e.g., pad/eos at 151643+). Take the true
    # max we'll see, rounded up to a multiple of 128 to match Qwen's actual
    # embedding-table size.
    max_id = int(shard.target_ids.max().item())
    needed = max(shard.vocab_size, max_id + 1, shard.pad_token_id + 1)
    cfg.voicebox.vocab_size = ((needed + 127) // 128) * 128

    projector = HyperProjector(cfg.projector, cfg.voicebox).to(device)
    voicebox = Voicebox(cfg.voicebox).to(device)
    optimizer = build_optimizer(projector, voicebox, cfg.train)
    return TrainState(
        projector=projector,
        voicebox=voicebox,
        optimizer=optimizer,
        pad_id=shard.pad_token_id,
        cfg=cfg,
        device=torch.device(device),
    )


# --- Train / eval steps ------------------------------------------------------

def train_step(state: TrainState, batch: tuple) -> dict:
    state.projector.train()
    state.voicebox.train()
    concept, target_ids, target_mask = (t.to(state.device, non_blocking=True) for t in batch)
    target_ids = target_ids.long()
    target_mask = target_mask.bool()

    lr = lr_at_step(state.step, state.cfg.train)
    for pg in state.optimizer.param_groups:
        pg["lr"] = lr

    loss, metrics = compute_loss(
        state.projector, state.voicebox, concept, target_ids, target_mask, state.pad_id
    )
    state.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(state.projector.parameters()) + list(state.voicebox.parameters()), 1.0
    )
    state.optimizer.step()
    state.step += 1
    metrics["lr"] = lr
    return metrics


@torch.no_grad()
def eval_loop(state: TrainState, loader: DataLoader) -> dict:
    state.projector.eval()
    state.voicebox.eval()
    totals = {"loss": 0.0, "token_acc": 0.0, "first_acc": 0.0}
    n = 0
    for batch in loader:
        concept, target_ids, target_mask = (t.to(state.device, non_blocking=True) for t in batch)
        target_ids = target_ids.long()
        target_mask = target_mask.bool()
        _, metrics = compute_loss(
            state.projector, state.voicebox, concept, target_ids, target_mask, state.pad_id
        )
        for k in totals:
            totals[k] += metrics[k]
        n += 1
    return {k: v / max(1, n) for k, v in totals.items()}


# --- Checkpoint --------------------------------------------------------------

def save_checkpoint(state: TrainState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "projector": state.projector.state_dict(),
            "voicebox": state.voicebox.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "step": state.step,
            "best_eval_loss": state.best_eval_loss,
            "voicebox_cfg": state.cfg.voicebox.__dict__,
            "projector_cfg": state.cfg.projector.__dict__,
            "pad_id": state.pad_id,
        },
        path,
    )


def load_checkpoint(path: Path, state: TrainState) -> None:
    blob = torch.load(path, map_location=state.device, weights_only=False)
    state.projector.load_state_dict(blob["projector"])
    state.voicebox.load_state_dict(blob["voicebox"])
    state.optimizer.load_state_dict(blob["optimizer"])
    state.step = blob["step"]
    state.best_eval_loss = blob["best_eval_loss"]
