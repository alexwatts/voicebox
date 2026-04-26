"""Post-filter and subsample an existing curated JSONL without re-running the
teacher. Use this after curate_facts.py to tighten quality cheaply.

What the post-filter can do (bounded by what's in the JSONL):
  * Enforce a minimum / maximum target length (in normalized chars).
  * Deduplicate by (prompt, target).
  * Subsample to a target size with a seeded RNG.

What it CANNOT do (those need the original generation text):
  * Word-boundary matching against the teacher's output.
  * Penalize hedging / preamble.

If you need that, re-run curate_facts.py with stricter rules.

Example:
    python scripts/subsample_facts.py \
        --in-train  data/raw/qwen_known.train.jsonl \
        --in-test   data/raw/qwen_known.test.jsonl \
        --out-train data/raw/qwen_known.tight.train.jsonl \
        --out-test  data/raw/qwen_known.tight.test.jsonl \
        --n 10000 \
        --min-target-chars 4 \
        --max-target-chars 60
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from voicebox.data import load_jsonl, normalize_answer, write_jsonl


def post_filter(
    records: list[dict],
    min_target_chars: int,
    max_target_chars: int,
    max_target_tokens: int = 0,
    tokenizer=None,
) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    n_drop_token_len = 0
    for r in records:
        prompt = r["prompt"].strip()
        target = r["target"]
        norm_target = normalize_answer(target)
        if len(norm_target) < min_target_chars:
            continue
        if len(norm_target) > max_target_chars:
            continue
        if max_target_tokens > 0 and tokenizer is not None:
            n_tok = len(tokenizer(target, add_special_tokens=False)["input_ids"])
            if n_tok > max_target_tokens:
                n_drop_token_len += 1
                continue
        key = (prompt, norm_target)
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    if n_drop_token_len:
        print(f"  dropped {n_drop_token_len} records exceeding "
              f"{max_target_tokens}-token target cap")
    return kept


def subsample(records: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or n >= len(records):
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-train", type=Path, required=True)
    p.add_argument("--in-test", type=Path, required=True)
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-test", type=Path, required=True)
    p.add_argument("--n", type=int, default=10000,
                   help="Target train size after filter+subsample. 0 = keep all.")
    p.add_argument("--test-n", type=int, default=0,
                   help="Optional cap on test size. 0 = keep all surviving.")
    p.add_argument("--min-target-chars", type=int, default=4)
    p.add_argument("--max-target-chars", type=int, default=60)
    p.add_argument("--max-target-tokens", type=int, default=0,
                   help="Drop records whose target exceeds this many tokens. "
                        "0 = disabled. Requires --tokenizer.")
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tokenizer = None
    if args.max_target_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer for token-count filter: {args.tokenizer}")

    train_in = load_jsonl(args.in_train)
    test_in = load_jsonl(args.in_test)
    print(f"Loaded train={len(train_in)} test={len(test_in)}")

    train_filtered = post_filter(
        train_in, args.min_target_chars, args.max_target_chars,
        args.max_target_tokens, tokenizer,
    )
    test_filtered = post_filter(
        test_in, args.min_target_chars, args.max_target_chars,
        args.max_target_tokens, tokenizer,
    )
    print(f"After filter: train={len(train_filtered)} test={len(test_filtered)}")

    train_out = subsample(train_filtered, args.n, args.seed)
    test_out = subsample(test_filtered, args.test_n, args.seed + 1)
    print(f"After subsample: train={len(train_out)} test={len(test_out)}")

    write_jsonl(train_out, args.out_train)
    write_jsonl(test_out, args.out_test)
    print(f"Wrote {args.out_train}")
    print(f"Wrote {args.out_test}")
    print("Sample kept records:")
    for r in train_out[:5]:
        print(f"  {r['prompt']!r} -> {r['target']!r}")


if __name__ == "__main__":
    main()
