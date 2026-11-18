import json
import random
from pathlib import Path

SEED = 20261118
INPUTS = [
    Path("data/raw/math_synth.jsonl"),
    Path("data/raw/word_problems.jsonl"),
    Path("data/raw/logic_synth.jsonl"),
]
OUTPUT = Path("data/raw/reasoning_synth.jsonl")


def main():
    rng = random.Random(SEED)
    rows = []
    counts = {}

    for path in INPUTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Generate all source datasets first.")
        count = 0
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if not isinstance(row.get("text"), str):
                    raise ValueError(f"Invalid text field in {path}")
                rows.append(row)
                count += 1
        counts[str(path)] = count

    rng.shuffle(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Input counts:")
    for path, count in counts.items():
        print(f"  {path}: {count:,}")
    print(f"Combined: {len(rows):,} examples -> {OUTPUT}")
    print("The seeded shuffle mixes categories throughout the file, including its tail.")


if __name__ == "__main__":
    main()
