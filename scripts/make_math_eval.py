import itertools
import json
import random
from pathlib import Path

random.seed(1234)

N_PER_CATEGORY = 20
OUT_PATH = Path("eval_data") / "math_eval.jsonl"

NAMES = ["Tom", "Sara", "Ali", "Maya", "Omar", "Lina"]
ITEMS = ["apples", "pens", "books", "balls", "cards", "stickers"]
KINDS = [("dog", "pet"), ("cat", "pet"), ("rose", "flower"), ("tulip", "flower"),
         ("car", "vehicle"), ("bus", "vehicle"), ("pear", "fruit"), ("plum", "fruit")]


def make_add(lo, hi):
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    return f"What is {a} + {b}?", str(a + b)


def make_sub(lo, hi):
    a, b = random.randint(lo, hi), random.randint(lo, hi)
    a, b = max(a, b), min(a, b)
    return f"What is {a} - {b}?", str(a - b)


def make_mul(lo_a, hi_a, lo_b, hi_b):
    a, b = random.randint(lo_a, hi_a), random.randint(lo_b, hi_b)
    return f"What is {a} * {b}?", str(a * b)


def make_compare():
    a, b = random.sample(range(10, 100), 2)
    return f"Which number is larger, {a} or {b}?", str(max(a, b))


def make_sequence():
    start, step = random.randint(1, 20), random.randint(2, 9)
    seq = [start + step * i for i in range(4)]
    nxt = start + step * 4
    return f"What comes next: {', '.join(map(str, seq))}, ?", str(nxt)


def make_word():
    name, item = random.choice(NAMES), random.choice(ITEMS)
    a = random.randint(5, 30)
    b = random.randint(1, a)
    if random.random() < 0.5:
        q = f"{name} has {a} {item}. {name} buys {b} more. How many {item} does {name} have now?"
        return q, str(a + b)
    q = f"{name} has {a} {item}. {name} gives away {b}. How many {item} does {name} have now?"
    return q, str(a - b)


_yes_no = itertools.cycle([True, False])


def make_logic():
    want_yes = next(_yes_no)
    name = random.choice(NAMES)
    (k1, g1), (k2, g2) = random.sample(KINDS, 2)
    while g1 == g2:
        (k1, g1), (k2, g2) = random.sample(KINDS, 2)
    if want_yes:
        q = f"All {k1}s are {g1}s. {name} is a {k1}. Is {name} a {g1}? Answer yes or no."
        return q, "yes"
    q = f"No {k1}s are {g2}s. {name} is a {k1}. Is {name} a {g2}? Answer yes or no."
    return q, "no"


CATEGORIES = {
    "add_1digit": lambda: make_add(1, 9),
    "add_2digit": lambda: make_add(10, 99),
    "add_3digit": lambda: make_add(100, 999),
    "sub_2digit": lambda: make_sub(10, 99),
    "mul_1digit": lambda: make_mul(2, 9, 2, 9),
    "mul_2x1": lambda: make_mul(10, 99, 2, 9),
    "compare": make_compare,
    "sequence": make_sequence,
    "word_problem": make_word,
    "logic_yes_no": make_logic,
}


def main():
    if OUT_PATH.exists():
        print(f"{OUT_PATH} already exists. Not overwriting the frozen eval set.")
        return
    OUT_PATH.parent.mkdir(exist_ok=True)

    rows = []
    for category, make_fn in CATEGORIES.items():
        seen = set()
        attempts = 0
        while len(seen) < N_PER_CATEGORY and attempts < 1000:
            attempts += 1
            question, answer = make_fn()
            if question in seen:
                continue
            seen.add(question)
            rows.append({"category": category, "question": question, "answer": answer})
        print(f"{category}: {len(seen)} problems")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} problems to {OUT_PATH}")


if __name__ == "__main__":
    main()