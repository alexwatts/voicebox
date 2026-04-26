"""Aha-moment evaluation.

Load a trained projector + voicebox checkpoint, run greedy autoregressive
generation on the held-out test shard, and report:
  * first-token accuracy
  * exact-match accuracy (after normalize_answer)
  * 'contains' accuracy (target appears in generation, normalized)
  * a sample of decoded generations vs. ground truth.

The teacher is NOT loaded — we use the saved test-shard concept vectors and
just need the teacher's tokenizer (downloaded fresh, tiny) to decode.

Example:
    python scripts/eval.py \
        --shard data/vectors/qwen25_7b.test.pt \
        --ckpt  checkpoints/voicebox.final.pt \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --max-new-tokens 16 \
        --n-samples 25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from voicebox.config import Config, ProjectorConfig, VoiceboxConfig
from voicebox.data import VectorShard, normalize_answer
from voicebox.projector import HyperProjector
from voicebox.voicebox import Voicebox


@torch.no_grad()
def generate_greedy(
    projector: HyperProjector,
    voicebox: Voicebox,
    concept: torch.Tensor,           # (B, teacher_hidden_dim)
    pad_id: int,
    max_new_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Greedy autoregressive decode. Returns (B, max_new_tokens)."""
    B = concept.size(0)
    pairs, bias = projector(concept)
    out_tokens = torch.full(
        (B, max_new_tokens), pad_id, dtype=torch.long, device=device
    )
    cur_input = torch.full((B, 1), pad_id, dtype=torch.long, device=device)
    for t in range(max_new_tokens):
        # Voicebox's internal pos_emb expects positions starting at 0; we
        # rebuild the running sequence each step so positions stay aligned
        # with what training saw.
        if t == 0:
            seq = cur_input
        else:
            seq = torch.cat([cur_input, out_tokens[:, :t]], dim=1)
        logits = voicebox(seq, pairs, concept_bias=bias)  # (B, T, V)
        next_tok = logits[:, -1].argmax(dim=-1)  # (B,)
        out_tokens[:, t] = next_tok
    return out_tokens


def reconstruct_modules(
    blob: dict, shard: VectorShard, device: torch.device
) -> tuple[HyperProjector, Voicebox]:
    """Rebuild projector + voicebox with the dimensions saved in the ckpt."""
    cfg = Config()
    cfg.voicebox.__dict__.update(blob["voicebox_cfg"])
    cfg.projector.__dict__.update(blob["projector_cfg"])
    cfg.projector.teacher_hidden_dim = shard.teacher_hidden_dim

    projector = HyperProjector(cfg.projector, cfg.voicebox).to(device)
    voicebox = Voicebox(cfg.voicebox).to(device)
    projector.load_state_dict(blob["projector"])
    voicebox.load_state_dict(blob["voicebox"])
    projector.eval()
    voicebox.eval()
    return projector, voicebox


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=25,
                   help="How many decoded examples to print.")
    p.add_argument("--n-eval", type=int, default=0,
                   help="Cap how many shard rows we score (0 = all).")
    args = p.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    shard = VectorShard.load(args.shard)
    n = len(shard.prompts) if args.n_eval == 0 else min(args.n_eval, len(shard.prompts))
    print(f"shard: {len(shard.prompts)} rows, scoring {n}")

    blob = torch.load(args.ckpt, map_location=device, weights_only=False)
    projector, voicebox = reconstruct_modules(blob, shard, device)
    pad_id = blob["pad_id"]
    print(f"loaded ckpt @ step {blob.get('step', '?')}, best_eval_loss={blob.get('best_eval_loss', '?')}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    n_first = n_exact = n_contains = 0
    samples: list[tuple[str, str, str]] = []  # (prompt, target, gen)

    for i in range(0, n, args.batch_size):
        idx = slice(i, min(i + args.batch_size, n))
        concept = shard.vectors[idx].to(device)
        target_ids = shard.target_ids[idx].to(device)
        target_mask = shard.target_mask[idx].to(device)

        gen = generate_greedy(projector, voicebox, concept, pad_id, args.max_new_tokens, device)

        # First-token accuracy.
        first_correct = (gen[:, 0] == target_ids[:, 0]) & target_mask[:, 0]
        n_first += int(first_correct.sum().item())

        # Decode each generation; compare normalized strings.
        gen_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for j, gtext in enumerate(gen_texts):
            row_i = i + j
            if row_i >= n:
                break
            target_text = shard.targets[row_i]
            norm_t = normalize_answer(target_text)
            norm_g = normalize_answer(gtext)
            if norm_t == norm_g:
                n_exact += 1
            if norm_t and norm_t in norm_g:
                n_contains += 1
            if len(samples) < args.n_samples:
                samples.append((shard.prompts[row_i], target_text, gtext))

    print()
    print(f"first_token_acc: {n_first / n:.4f}  ({n_first}/{n})")
    print(f"exact_match:     {n_exact / n:.4f}  ({n_exact}/{n})")
    print(f"contains:        {n_contains / n:.4f}  ({n_contains}/{n})")
    print()
    print("Samples (prompt | target | generated):")
    for prompt, target, gen in samples:
        print(f"  Q: {prompt!r}")
        print(f"     target:    {target!r}")
        print(f"     generated: {gen!r}")
        print()


if __name__ == "__main__":
    main()
