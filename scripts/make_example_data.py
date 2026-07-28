#!/usr/bin/env python3
"""Generate a small synthetic dataset for smoke-testing both pipelines.

The text is assembled from fixed templates with a fixed random seed. It is
NOT real model output and contains no real user data: its only purpose is to
let someone clone the repository and execute both pipelines end to end before
supplying their own data.

    python scripts/make_example_data.py --output data/example/responses_example.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

MODELS = ["model_a", "model_b"]
PROMPTS = [1, 2, 3]
GROUPS = ["base", "christian", "jewish", "muslim", "male", "female"]
N_PER_CELL = 6

OPENINGS = [
    "That sounds like a really difficult situation to be in.",
    "It's understandable that you're feeling unsettled by this.",
    "Thank you for sharing something so personal.",
    "This is a common concern and it's worth thinking through carefully.",
]

MIDDLES = [
    "What they did was disrespectful and shows a lack of consideration for you.",
    "It might help to sit down and talk it through when you're both calm.",
    "This is a red flag and it could escalate if nothing changes.",
    "You handled that with a lot of patience and self-control.",
    "Consider whether this is part of a wider pattern of controlling behaviour.",
    "Setting a clear boundary here would be reasonable and fair.",
]

CLOSINGS = [
    "Here are some steps: first write down how you feel, then decide what you want, then say it plainly.",
    "Speaking to a therapist or counselor could give you a neutral space to work through it.",
    "You deserve to be treated with respect, and it's okay to walk away if that doesn't happen.",
    "Give it some time and see whether they are willing to make a real change.",
    "If you ever feel unsafe, contact a domestic violence hotline or the police.",
]

RELIGIOUS_CLOSINGS = {
    "christian": "Pray about it, and trust that God has a plan for your marriage.",
    "jewish": "Speaking with your rabbi may help you find clarity through the wisdom of Torah.",
    "muslim": "Make istikhara and seek guidance from your imam before you decide.",
}


def build_row(rng: random.Random, model: str, prompt: int, group: str, index: int) -> dict:
    parts = [
        rng.choice(OPENINGS),
        rng.choice(MIDDLES),
        rng.choice(MIDDLES),
        rng.choice(CLOSINGS),
    ]
    if group in RELIGIOUS_CLOSINGS and rng.random() < 0.7:
        parts.append(RELIGIOUS_CLOSINGS[group])
    return {
        "ID": f"{model}_p{prompt}_{group}_{index:02d}",
        "model": model,
        "prompt_number": prompt,
        "identity": group,
        "text": " ".join(parts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/example/responses_example.csv")
    parser.add_argument("--seed", type=int, default=20240101)
    parser.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [
        build_row(rng, model, prompt, group, i)
        for model in MODELS
        for prompt in PROMPTS
        for group in GROUPS
        for i in range(args.n_per_cell)
    ]

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Wrote {len(rows)} synthetic rows to {path}")


if __name__ == "__main__":
    main()
