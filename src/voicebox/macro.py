"""Teacher (macromodel) loading and concept-vector extraction.

The teacher is frozen; we only ever run forward passes through it. We grab the
last hidden state of the final non-padding token as the "concept vector" that
the hyper-projector will turn into voicebox weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

from .config import TeacherConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedTeacher:
    model: AutoModel
    tokenizer: AutoTokenizer
    hidden_dim: int
    device: torch.device


def load_teacher(cfg: TeacherConfig) -> LoadedTeacher:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": _DTYPES[cfg.dtype], "device_map": cfg.device_map}
    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_DTYPES[cfg.dtype],
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)

    model = AutoModel.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = model.config.hidden_size
    device = next(model.parameters()).device
    return LoadedTeacher(model=model, tokenizer=tokenizer, hidden_dim=hidden_dim, device=device)


@torch.no_grad()
def extract_concept_vectors(
    teacher: LoadedTeacher,
    prompts: list[str],
    batch_size: int = 8,
    max_length: int = 256,
) -> torch.Tensor:
    """Return tensor of shape (len(prompts), hidden_dim) on CPU in float32."""
    out: list[torch.Tensor] = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = teacher.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(teacher.device)

        hs = teacher.model(**enc, output_hidden_states=False).last_hidden_state
        # Pick the last non-pad token per sequence.
        last_idx = enc.attention_mask.sum(dim=1) - 1  # (B,)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
        last_tok = hs.gather(1, gather_idx).squeeze(1)  # (B, H)
        out.append(last_tok.detach().to("cpu", dtype=torch.float32))

    return torch.cat(out, dim=0)
