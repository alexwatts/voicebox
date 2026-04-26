"""Datasets for the JIT-compilation experiment.

We support two paradigms (see plan):

* **Paradigm B (active)** — facts the teacher actually knows, curated by
  filtering a public QA corpus (TriviaQA) through the teacher itself. Built by
  ``scripts/curate_facts.py``; loaded here via ``load_jsonl`` /
  ``train_test_split``.

* **Paradigm A (kept as a smoke-test fallback)** — synthetic fictional facts
  via ``generate()`` and the templates below. Useful for verifying the
  training plumbing without running the full curation pass. The label-noisy
  pet template has been removed so the remaining templates are bijective.
"""
from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch.utils.data import Dataset


# --- Word banks ---------------------------------------------------------------

PLANET_PREFIX = ["Xy", "Glo", "Zar", "Vee", "Tho", "Quel", "Mor", "Bri", "Nax", "Plu", "Kry", "Ven"]
PLANET_SUFFIX = ["lar", "rp", "ix", "non", "rin", "dor", "tos", "ven", "max", "is", "lon", "rax"]
CITY_PREFIX = ["Zo", "Blo", "Tre", "Quar", "Vex", "Pla", "Mor", "Drin", "Sko", "Yel", "Hax", "Bre"]
CITY_SUFFIX = ["g", "op", "th", "ix", "or", "an", "us", "in", "el", "om", "ar", "und"]

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Gina", "Hank",
         "Ivy", "Jack", "Kara", "Liam", "Mia", "Noah", "Owen", "Pia"]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
HUNDREDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

ANIMALS = ["fox", "owl", "cat", "wolf", "bear", "hawk", "lynx", "deer", "otter", "raven"]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "pink", "teal"]


# --- Template families --------------------------------------------------------

def _planet_capital(rng: random.Random) -> tuple[str, str]:
    planet = rng.choice(PLANET_PREFIX) + rng.choice(PLANET_SUFFIX)
    capital = rng.choice(CITY_PREFIX) + rng.choice(CITY_SUFFIX)
    return f"The capital of planet {planet} is", f" {capital}."


def _account_balance(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(NAMES) + "_" + rng.choice(NAMES)
    amount = rng.choice(HUNDREDS)
    return (
        f"The user {name} has an account balance of",
        f" {amount} hundred dollars.",
    )


def _favorite_color(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(NAMES)
    color = rng.choice(COLORS)
    return f"{name}'s favorite color is", f" {color}."


TEMPLATES: list[Callable[[random.Random], tuple[str, str]]] = [
    _planet_capital,
    _account_balance,
    _favorite_color,
]


# --- Generation API -----------------------------------------------------------

def generate(n: int, seed: int = 0) -> list[dict]:
    """Return n records of {'prompt': ..., 'target': ...} with deduping."""
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        tmpl = rng.choice(TEMPLATES)
        prompt, target = tmpl(rng)
        if (prompt, target) in seen:
            continue
        seen.add((prompt, target))
        out.append({"prompt": prompt, "target": target})
    if len(out) < n:
        raise RuntimeError(
            f"Could only generate {len(out)} unique records, needed {n}. "
            "Expand the word banks."
        )
    return out


def write_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def train_test_split(
    records: list[dict], test_frac: float, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Random shuffle + split. Returns (train, test)."""
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_frac))
    return shuffled[:cut], shuffled[cut:]


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Lower, drop punctuation, collapse whitespace.

    Used on both sides of the teacher-knowledge filter and any later
    string-match evaluation, so the comparison is consistent.
    """
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# --- PyTorch Dataset ----------------------------------------------------------

@dataclass
class VectorShard:
    """In-memory view of a saved extract_vectors.py shard."""
    vectors: torch.Tensor          # (N, teacher_hidden_dim) float32
    target_ids: torch.Tensor       # (N, T) long, padded with pad_token_id
    target_mask: torch.Tensor      # (N, T) bool, True where real token
    prompts: list[str]
    targets: list[str]
    vocab_size: int
    pad_token_id: int
    teacher_hidden_dim: int
    model_id: str

    @classmethod
    def load(cls, path: str | Path) -> "VectorShard":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            vectors=blob["vectors"],
            target_ids=blob["target_ids"],
            target_mask=blob["target_mask"],
            prompts=blob["prompts"],
            targets=blob["targets"],
            vocab_size=blob["vocab_size"],
            pad_token_id=blob["pad_token_id"],
            teacher_hidden_dim=blob["hidden_dim"],
            model_id=blob["model_id"],
        )


class VectorDataset(Dataset):
    """Yields (concept_vector, target_ids, target_mask) per item."""

    def __init__(self, shard: VectorShard):
        self.shard = shard

    def __len__(self) -> int:
        return self.shard.vectors.size(0)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.shard.vectors[i],
            self.shard.target_ids[i],
            self.shard.target_mask[i],
        )
