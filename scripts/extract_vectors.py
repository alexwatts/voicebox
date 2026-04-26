"""Phase 1: extract concept vectors from a frozen teacher.

Reads prompts from a JSONL file with one {"prompt": "...", "target": "..."}
record per line, runs them through the teacher, and saves the resulting
concept vectors plus the matching prompts/targets to a single .pt shard.

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
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

    # Chunk for progress bar.
    chunk = 64
    pieces = []
    for i in tqdm(range(0, len(prompts), chunk), desc="extract"):
        batch_prompts = prompts[i : i + chunk]
        v = extract_concept_vectors(
            teacher,
            batch_prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        pieces.append(v)
    vectors = torch.cat(pieces, dim=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vectors": vectors,
            "prompts": prompts,
            "targets": targets,
            "model_id": args.model,
            "hidden_dim": teacher.hidden_dim,
        },
        args.out,
    )
    print(f"Wrote {vectors.shape} -> {args.out}")


if __name__ == "__main__":
    main()
