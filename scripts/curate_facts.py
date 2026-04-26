"""Build a Paradigm-B fact dataset by filtering TriviaQA through the teacher.

Loop:
    1. Stream candidates from TriviaQA (rc.nocontext) — (question, aliases).
    2. Greedy-decode N tokens from the teacher with the question as prompt.
    3. Keep iff any answer alias appears in the decoded text after
       normalization.
    4. Write kept records as {"prompt": <question>, "target": " <answer>."}
       split into train/test JSONL files.

Example:
    python scripts/curate_facts.py \
        --out-train data/raw/qwen_known.train.jsonl \
        --out-test  data/raw/qwen_known.test.jsonl \
        --max-candidates 95000 \
        --batch-size 8 --dtype bfloat16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from voicebox.config import TeacherConfig
from voicebox.data import normalize_answer, train_test_split, write_jsonl
from voicebox.macro import load_teacher


def iter_trivia_candidates(max_candidates: int):
    """Yield (question: str, aliases: list[str]) tuples from TriviaQA."""
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_candidates:
            break
        q = row["question"].strip()
        aliases = row["answer"]["aliases"] + [row["answer"]["value"]]
        aliases = [a for a in (a.strip() for a in aliases) if a]
        if not aliases:
            continue
        yield q, aliases


@torch.no_grad()
def teacher_answers(teacher, questions: list[str], max_new_tokens: int) -> list[str]:
    """Greedy-decode the teacher on a batch of questions, return generated text."""
    enc = teacher.tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(teacher.device)

    # AutoModel doesn't have a generation head; we need the LM. Reload-once
    # check: caller is expected to pass a model that supports .generate().
    out = teacher.model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=teacher.tokenizer.pad_token_id,
    )
    # Slice off the prompt tokens to get just the continuation.
    new_tokens = out[:, enc.input_ids.size(1) :]
    return teacher.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


_MIN_ALIAS_CHARS = 3


def is_correct(generated: str, aliases: list[str]) -> str | None:
    """Return the matching alias if any alias appears in the generation,
    after normalization. Otherwise None.

    Aliases shorter than _MIN_ALIAS_CHARS are skipped to avoid spurious
    matches (e.g., a "US" alias matching the generated word "trust").
    """
    norm_gen = normalize_answer(generated)
    if not norm_gen:
        return None
    for alias in aliases:
        norm_alias = normalize_answer(alias)
        if len(norm_alias) < _MIN_ALIAS_CHARS:
            continue
        if norm_alias in norm_gen:
            return alias
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-train", type=Path, required=True)
    p.add_argument("--out-test", type=Path, required=True)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--max-candidates", type=int, default=95000)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    args = p.parse_args()

    cfg = TeacherConfig(
        model_id=args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        load_in_4bit=args.load_in_4bit,
    )
    teacher = load_teacher_for_generation(cfg)
    print(f"Loaded teacher {args.model} on {teacher.device}")

    kept: list[dict] = []
    seen_questions: set[str] = set()
    n_seen = 0

    pbar = tqdm(total=args.max_candidates, desc="filter")
    batch_q: list[str] = []
    batch_aliases: list[list[str]] = []

    def flush_batch():
        nonlocal kept, batch_q, batch_aliases
        if not batch_q:
            return
        outs = teacher_answers(teacher, batch_q, args.max_new_tokens)
        for q, aliases, gen in zip(batch_q, batch_aliases, outs):
            match = is_correct(gen, aliases)
            if match is not None:
                target = f" {match}."
                kept.append({"prompt": q, "target": target})
        batch_q.clear()
        batch_aliases.clear()

    for q, aliases in iter_trivia_candidates(args.max_candidates):
        n_seen += 1
        pbar.update(1)
        if q in seen_questions:
            continue
        seen_questions.add(q)
        batch_q.append(q)
        batch_aliases.append(aliases)
        if len(batch_q) >= args.batch_size:
            flush_batch()
            pbar.set_postfix(kept=len(kept), rate=f"{len(kept)/max(n_seen,1):.0%}")
    flush_batch()
    pbar.close()

    print(f"Saw {n_seen} candidates, kept {len(kept)} ({len(kept)/max(n_seen,1):.1%}).")
    train, test = train_test_split(kept, test_frac=args.test_frac, seed=args.seed)
    write_jsonl(train, args.out_train)
    write_jsonl(test, args.out_test)
    print(f"Wrote {len(train)} -> {args.out_train}")
    print(f"Wrote {len(test)} -> {args.out_test}")
    if kept:
        print("Sample kept records:")
        for r in kept[:5]:
            print(f"  {r['prompt']!r} -> {r['target']!r}")


# --- Loader variant that returns a model with .generate() -------------------

def load_teacher_for_generation(cfg: TeacherConfig):
    """Like voicebox.macro.load_teacher but uses AutoModelForCausalLM so we
    have an LM head available for .generate(). Hidden states are still
    accessible via output_hidden_states=True if we wanted them later."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from voicebox.macro import LoadedTeacher, _DTYPES

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left-pad for generation so the last token of every sequence aligns.
    tokenizer.padding_side = "left"

    kwargs = {"torch_dtype": _DTYPES[cfg.dtype], "device_map": cfg.device_map}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPES[cfg.dtype],
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    return LoadedTeacher(model=model, tokenizer=tokenizer, hidden_dim=hidden_dim, device=device)


if __name__ == "__main__":
    main()
