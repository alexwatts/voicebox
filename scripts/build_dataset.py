"""Generate the synthetic Phase-3 dataset and write it to JSONL.

Example:
    python scripts/build_dataset.py --n 10000 --out data/raw/synthetic.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from voicebox.data import generate, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/raw/synthetic.jsonl"))
    args = p.parse_args()

    records = generate(args.n, seed=args.seed)
    write_jsonl(records, args.out)
    print(f"Wrote {len(records)} records -> {args.out}")
    print("Sample:")
    for r in records[:5]:
        print(f"  {r['prompt']!r} -> {r['target']!r}")


if __name__ == "__main__":
    main()
