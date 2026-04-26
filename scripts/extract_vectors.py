"""Phase 1: extract concept vectors from a frozen teacher AND tokenize targets
with the teacher's tokenizer (so the voicebox shares its vocab).

Reads prompts from a JSONL file with one {"prompt": "...", "target": "..."}
record per line. Writes a single .pt shard containing:
    vectors        (N, teacher_hidden_dim) float32
    target_ids     (N, T) long, right-padded with pad_token_id
    target_mask    (N, T) bool, True where the position is a real target token
    prompts        list[str]
    targets        list[str]
    vocab_size     int
    pad_token_id   int
    hidden_dim     int
    model_id       str

Example:
  python scripts/extract_vectors.py \
      --prompts data/raw/synthetic.jsonl \
      --out data/vectors/qwen25_7b.pt \
      --model Qwen/Qwen2.5-7B-Instruct \
      --batch-size 8 --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from voicebox.config import TeacherConfig
from voicebox.macro import extract_concept_vectors, load_teacher


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_targets(
    tokenizer, targets: list[str], max_target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad target token IDs to a common length. Returns (ids, mask)."""
    enc = tokenizer(
        targets,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_target_len,
        add_special_tokens=False,  # targets are continuations, not standalone seqs
    )
    return enc["input_ids"].long(), enc["attention_mask"].bool()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-target-len", type=int, default=16)
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    records = read_jsonl(args.prompts)
    prompts = [r["prompt"] for r in records]
    targets = [r.get("target", "") for r in records]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")

    cfg = TeacherConfig(
        model_id=args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_length,
        load_in_4bit=args.load_in_4bit,
    )
    teacher = load_teacher(cfg)
    print(f"Loaded teacher {args.model} on {teacher.device} (hidden_dim={teacher.hidden_dim})")

    # 1. Concept vectors.
    chunk = 64
    pieces = []
    for i in tqdm(range(0, len(prompts), chunk), desc="extract"):
        v = extract_concept_vectors(
            teacher,
            prompts[i : i + chunk],
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        pieces.append(v)
    vectors = torch.cat(pieces, dim=0)

    # 2. Target token IDs (using the teacher's tokenizer).
    target_ids, target_mask = tokenize_targets(teacher.tokenizer, targets, args.max_target_len)
    truncated = (target_mask.sum(dim=1) == args.max_target_len).sum().item()
    if truncated:
        print(f"warning: {truncated} targets hit the {args.max_target_len}-token cap")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": vectors,
            "target_ids": target_ids,
            "target_mask": target_mask,
            "prompts": prompts,
            "targets": targets,
            "vocab_size": int(teacher.tokenizer.vocab_size),
            "pad_token_id": int(teacher.tokenizer.pad_token_id),
            "hidden_dim": teacher.hidden_dim,
            "model_id": args.model,
        },
        args.out,
    )
    print(
        f"Wrote vectors={tuple(vectors.shape)} target_ids={tuple(target_ids.shape)} "
        f"vocab={teacher.tokenizer.vocab_size} -> {args.out}"
    )


if __name__ == "__main__":
    main()
