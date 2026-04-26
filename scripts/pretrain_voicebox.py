"""Pre-train the voicebox as a standalone causal LM on a text corpus.

The point: give the voicebox basic language-model priors (token semantics,
n-gram statistics, how to start/stop a sentence) BEFORE we ask it to use the
LoRA + concept-bias modulation to articulate facts. Per OBJECTIVE.MD Phase 5,
this is the difference between "tiny model also has to learn English from
scratch" and "tiny model only has to learn how to route teacher knowledge."

The teacher is not loaded — only its tokenizer, to keep the vocab consistent
with the downstream training run.

Example:
    python scripts/pretrain_voicebox.py \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --dataset wikitext --dataset-config wikitext-103-raw-v1 \
        --out checkpoints/voicebox.pretrained.pt \
        --steps 8000 --batch-size 64 --seq-len 64 --lr 3e-4
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from voicebox.config import Config
from voicebox.voicebox import Voicebox


def stream_packed_chunks(tokenizer, dataset_iter, seq_len: int, target_tokens: int):
    """Tokenize streaming text and pack into fixed-length chunks. Returns a
    1-D long tensor of length ~target_tokens (we slice it into seq_len rows
    later)."""
    buf: list[int] = []
    for row in dataset_iter:
        text = row.get("text") or ""
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        buf.append(tokenizer.eos_token_id)
        if len(buf) >= target_tokens:
            break
    n_chunks = len(buf) // seq_len
    arr = torch.tensor(buf[: n_chunks * seq_len], dtype=torch.long)
    return arr.view(n_chunks, seq_len)


def lr_at_step(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset", type=str, default="wikitext")
    p.add_argument("--dataset-config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--target-tokens", type=int, default=80_000_000,
                   help="How many tokens to pull from the corpus before stopping.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use the model.config.vocab_size convention (rounded up) — matches what
    # extract_vectors.py and train.py do, so the pretrained voicebox plugs
    # into the JIT pipeline without an embedding-shape mismatch.
    base_vocab = tokenizer.vocab_size
    needed = max(base_vocab, tokenizer.pad_token_id + 1, tokenizer.eos_token_id + 1)
    vocab_size = ((needed + 127) // 128) * 128

    cfg = Config()
    cfg.voicebox.vocab_size = vocab_size
    cfg.voicebox.n_layers = args.n_layers
    cfg.voicebox.n_heads = args.n_heads
    cfg.voicebox.d_model = args.d_model
    cfg.voicebox.max_seq_len = max(cfg.voicebox.max_seq_len, args.seq_len)

    voicebox = Voicebox(cfg.voicebox).to(device)
    n_params = sum(p.numel() for p in voicebox.parameters())
    print(f"voicebox params: {n_params:,}  (vocab={vocab_size})")

    # Stream + pack the corpus.
    print(f"loading dataset {args.dataset}/{args.dataset_config}...")
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split, streaming=True)
    chunks = stream_packed_chunks(tokenizer, ds, args.seq_len, args.target_tokens)
    print(f"packed {chunks.size(0):,} chunks of length {args.seq_len} "
          f"(~{chunks.numel():,} tokens)")
    chunks = chunks.to(device)

    optimizer = torch.optim.AdamW(
        voicebox.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    voicebox.train()
    t0 = time.time()
    for step in range(args.steps):
        idx = torch.randint(0, chunks.size(0), (args.batch_size,), device=device)
        batch = chunks[idx]                       # (B, T)
        inputs = batch[:, :-1]
        labels = batch[:, 1:]

        lr = lr_at_step(step, args.steps, args.warmup, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        logits = voicebox(inputs)                 # standalone LM mode
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(voicebox.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                tok_acc = (preds == labels).float().mean().item()
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / max(1e-6, elapsed)
            print(
                f"step={step:5d}  loss={loss.item():.4f}  ppl={math.exp(loss.item()):.1f}  "
                f"tok_acc={tok_acc:.3f}  lr={lr:.2e}  {steps_per_sec:.1f} steps/s"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "voicebox": voicebox.state_dict(),
            "voicebox_cfg": cfg.voicebox.__dict__,
        },
        args.out,
    )
    print(f"wrote pretrained voicebox -> {args.out}")


if __name__ == "__main__":
    main()
