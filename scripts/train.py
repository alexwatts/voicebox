"""Train the projector + voicebox on a saved vector shard.

The teacher is not loaded here — its hidden states are already in the shard.

Example:
    python scripts/train.py \
        --shard data/vectors/qwen25_7b.train.pt \
        --out   checkpoints/voicebox.pt \
        --steps 5000 \
        --batch-size 32 \
        --lr 3e-4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from voicebox.config import Config
from voicebox.data import VectorDataset, VectorShard
from voicebox.train import (
    eval_loop,
    make_state_from_shard,
    save_checkpoint,
    train_step,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("checkpoints/voicebox.pt"))
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-frac", type=float, default=0.10)
    p.add_argument("--lora-rank", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}")

    shard = VectorShard.load(args.shard)
    print(
        f"shard: vectors={tuple(shard.vectors.shape)} "
        f"target_ids={tuple(shard.target_ids.shape)} "
        f"hidden_dim={shard.teacher_hidden_dim} vocab={shard.vocab_size}"
    )

    cfg = Config()
    cfg.voicebox.lora_rank = args.lora_rank
    cfg.voicebox.n_layers = args.n_layers
    cfg.voicebox.d_model = args.d_model
    cfg.voicebox.n_heads = args.n_heads
    cfg.voicebox.max_seq_len = max(cfg.voicebox.max_seq_len, shard.target_ids.size(1))
    cfg.train.lr = args.lr
    cfg.train.weight_decay = args.weight_decay
    cfg.train.n_steps = args.steps
    cfg.train.warmup_steps = args.warmup
    cfg.train.batch_size = args.batch_size

    state = make_state_from_shard(shard, cfg, device)
    p_params = sum(p.numel() for p in state.projector.parameters())
    v_params = sum(p.numel() for p in state.voicebox.parameters())
    print(f"projector params: {p_params:,}    voicebox params: {v_params:,}")

    full = VectorDataset(shard)
    n_eval = max(1, int(len(full) * args.eval_frac))
    n_train = len(full) - n_eval
    train_ds, eval_ds = random_split(
        full, [n_train, n_eval], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=(device == "cuda"))
    print(f"train={n_train} eval={n_eval}")

    train_iter = iter(train_loader)
    t0 = time.time()
    while state.step < args.steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        m = train_step(state, batch)

        if state.step % args.log_every == 0 or state.step == 1:
            steps_per_sec = state.step / max(1e-6, time.time() - t0)
            print(
                f"step={state.step:5d}  loss={m['loss']:.4f}  "
                f"tok_acc={m['token_acc']:.3f}  first_acc={m['first_acc']:.3f}  "
                f"lr={m['lr']:.2e}  {steps_per_sec:.1f} steps/s"
            )

        if state.step % args.eval_every == 0:
            ev = eval_loop(state, eval_loader)
            tag = ""
            if ev["loss"] < state.best_eval_loss:
                state.best_eval_loss = ev["loss"]
                save_checkpoint(state, args.out)
                tag = "  *saved*"
            print(
                f"  EVAL@{state.step}  loss={ev['loss']:.4f}  "
                f"tok_acc={ev['token_acc']:.3f}  first_acc={ev['first_acc']:.3f}{tag}"
            )

    # Final eval + save.
    ev = eval_loop(state, eval_loader)
    print(f"FINAL  loss={ev['loss']:.4f}  tok_acc={ev['token_acc']:.3f}  first_acc={ev['first_acc']:.3f}")
    save_checkpoint(state, args.out.with_suffix(".final.pt"))
    print(f"wrote final ckpt -> {args.out.with_suffix('.final.pt')}")
    print(f"best eval ckpt    -> {args.out}  (loss={state.best_eval_loss:.4f})")


if __name__ == "__main__":
    main()
