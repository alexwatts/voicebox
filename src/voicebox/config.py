"""Hyperparameter dataclasses. Defaults target Colab Pro (L4 / A100, bf16)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeacherConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    max_prompt_tokens: int = 256
    batch_size: int = 8


@dataclass
class VoiceboxConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    max_seq_len: int = 64
    # Placeholder default — override from the extracted shard at train time
    # (shard["vocab_size"]) so the voicebox shares the teacher's tokenizer.
    vocab_size: int = 32000
    lora_rank: int = 16
    dropout: float = 0.0


@dataclass
class ProjectorConfig:
    teacher_hidden_dim: int = 3584  # Qwen2.5-7B
    hidden_dim: int = 1024
    n_hidden_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 0
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    n_steps: int = 5000
    warmup_steps: int = 200
    log_every: int = 50
    eval_every: int = 500
    ckpt_dir: str = "checkpoints"
    vectors_dir: str = "data/vectors"
    prompts_path: str = "data/raw/synthetic.jsonl"


@dataclass
class Config:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    voicebox: VoiceboxConfig = field(default_factory=VoiceboxConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
