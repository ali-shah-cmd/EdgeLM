import argparse
import json
import random
from pathlib import Path

# Stage-2 reasoning mix: arith_v2 + logic_v2 + a subset of the existing word problems.
# Run from the project root (C:\Work\SLM). Reads three files, writes one new file.

parser = argparse.ArgumentParser()
parser.add_argument("--arith", default="data/raw/arith_v2.jsonl")
parser.add_argument("--logic", default="data/raw/logic_v2.jsonl")
parser.add_argument("--word", default="data/raw/word_problems.jsonl")
parser.add_argument("--n_arith", type=int, default=500_000)
parser.add_argument("--n_logic", type=int, default=500_000)
parser.add_argument("--n_word", type=int, default=500_000)
parser.add_argument("--out", default="data/raw/reasoning_v2.jsonl")
parser.add_argument("--seed", type=int, default=20261122)
args = parser.parse_args()

SOURCES = [
    ("arith_v2", Path(args.arith), args.n_arith),
    ("logic_v2", Path(args.logic), args.n_logic),
    ("word_problems", Path(args.word), args.n_word),
]

rng = random.Random(args.seed)
rows = []
counts = {}

for name, path, limit in SOURCES:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    taken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if taken >= limit:
                break
            row = json.loads(line)
            if not isinstance(row.get("text"), str):
                raise ValueError(f"Invalid text field in {path}")
            rows.append(row)
            taken += 1
    counts[name] = taken

# Shuffle so every kind of example appears everywhere in the file,
# including the last part that train_cont_reason.py reserves for validation.
rng.shuffle(rows)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Examples taken per source:")
for name, count in counts.items():
    print(f"  {name}: {count:,}")
print(f"Combined: {len(rows):,} examples -> {out_path}")