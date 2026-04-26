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
from pathlib import Path

import torch
from tqdm import tqdm

from voicebox.config import TeacherConfig
from voicebox.data import load_jsonl
from voicebox.macro import extract_concept_vectors, load_teacher


def tokenize_targets(
    tokenizer, targets: list[str], max_target_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize each target, append EOS, right-pad to max_target_len.
    Returns (ids, mask). EOS is included in the mask so the voicebox is
    trained to emit it at end-of-answer — that's the stop signal eval relies on.
    """
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    raw = tokenizer(
        targets,
        padding=False,
        truncation=True,
        max_length=max_target_len - 1,  # leave room for EOS
        add_special_tokens=False,
    )["input_ids"]

    B = len(raw)
    ids = torch.full((B, max_target_len), pad_id, dtype=torch.long)
    mask = torch.zeros((B, max_target_len), dtype=torch.bool)
    for i, toks in enumerate(raw):
        seq = toks + [eos_id]
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, : len(seq)] = True
    return ids, mask


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
    p.add_argument("--prompt-template", type=str, default="Q: {prompt}\nA:",
                   help="Wrap each prompt before extraction. The trailing "
                        "':' makes the last hidden state encode the answer's "
                        "first-token distribution. Use '{prompt}' to disable wrapping.")
    p.add_argument("--pool-last-k", type=int, default=4,
                   help="Mean-pool the last K real-token hidden states. "
                        "1 = original behavior (last token only).")
    args = p.parse_args()

    records = load_jsonl(args.prompts)
    raw_prompts = [r["prompt"] for r in records]
    prompts = [args.prompt_template.format(prompt=p) for p in raw_prompts]
    targets = [r.get("target", "") for r in records]
    print(f"Loaded {len(prompts)} prompts from {args.prompts}")
    print(f"prompt_template={args.prompt_template!r}  pool_last_k={args.pool_last_k}")

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
            pool_last_k=args.pool_last_k,
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
            "prompts": raw_prompts,
            "prompt_template": args.prompt_template,
            "targets": targets,
            # NOTE: use model.config.vocab_size (true embedding-table rows),
            # not tokenizer.vocab_size — Qwen and others tack added special
            # tokens onto IDs beyond the base BPE vocab.
            "vocab_size": int(teacher.model.config.vocab_size),
            "eos_token_id": int(teacher.tokenizer.eos_token_id),
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
