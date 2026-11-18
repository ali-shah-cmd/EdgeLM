import json
import random
from pathlib import Path

random.seed(777)  # different seed from the eval set

N_EXAMPLES = 2_000_000
EVAL_PATH = Path("eval_data/math_eval.jsonl")
OUT_PATH = Path("data/raw/math_synth.jsonl")


def place_values(n, width):
    # 347 with width 3 -> [300, 40, 7]
    s = str(n).zfill(width)
    return [int(d) * 10 ** (width - i - 1) for i, d in enumerate(s)]


def rand_num(max_digits):
    digits = random.randint(1, max_digits)
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    return random.randint(lo, 10 ** digits - 1)


def gen_add():
    a, b = rand_num(3), rand_num(3)
    expr = f"{a} + {b}"
    canon = f"What is {expr}?"
    q = random.choice([canon, canon, f"Compute {expr}.", f"Add {a} and {b}."])
    work = None
    w = max(len(str(a)), len(str(b)))
    if w > 1:
        pa, pb = place_values(a, w), place_values(b, w)
        pairs = " + ".join(f"({x} + {y})" for x, y in zip(pa, pb))
        sums = " + ".join(str(x + y) for x, y in zip(pa, pb))
        work = f"{expr} = {pairs} = {sums} = {a + b}"
    return dict(canon=canon, q=q, a=str(a + b), expr=expr, work=work)


def gen_sub():
    a, b = rand_num(3), rand_num(3)
    a, b = max(a, b), min(a, b)
    expr = f"{a} - {b}"
    canon = f"What is {expr}?"
    q = random.choice([canon, canon, f"Compute {expr}.", f"Subtract {b} from {a}."])
    parts = [p for p in place_values(b, len(str(b))) if p > 0]
    work = None
    if len(parts) > 1:
        chain = f"{a} - " + " - ".join(map(str, parts))
        cur = a - parts[0]
        rest = parts[1:]
        steps = []
        while rest:
            steps.append(f"{cur} - " + " - ".join(map(str, rest)))
            cur -= rest[0]
            rest = rest[1:]
        steps.append(str(cur))
        work = f"{expr} = {chain} = " + " = ".join(steps)
    return dict(canon=canon, q=q, a=str(a - b), expr=expr, work=work)


def gen_mul():
    a, b = rand_num(3), random.randint(2, 9)
    expr = f"{a} * {b}"
    canon = f"What is {expr}?"
    q = random.choice([canon, canon, f"Compute {expr}.", f"Multiply {a} by {b}."])
    parts = [p for p in place_values(a, len(str(a))) if p > 0]
    work = None
    if len(parts) > 1:
        left = " + ".join(f"({p} * {b})" for p in parts)
        right = " + ".join(str(p * b) for p in parts)
        work = f"{expr} = {left} = {right} = {a * b}"
    return dict(canon=canon, q=q, a=str(a * b), expr=expr, work=work)


def gen_compare():
    a, b = rand_num(3), rand_num(3)
    while a == b:
        b = rand_num(3)
    hi, lo = max(a, b), min(a, b)
    if random.random() < 0.7:
        canon, ans = f"Which number is larger, {a} or {b}?", hi
    else:
        canon, ans = f"Which number is smaller, {a} or {b}?", lo
    work = f"{hi} is greater than {lo}"
    return dict(canon=canon, q=canon, a=str(ans), expr=None, work=work)


def gen_sequence():
    start, step = random.randint(1, 50), random.randint(2, 12)
    n_shown = random.randint(4, 5)
    seq = [start + step * i for i in range(n_shown)]
    nxt = start + step * n_shown
    canon = f"What comes next: {', '.join(map(str, seq))}, ?"
    work = f"each number goes up by {step}, so after {seq[-1]} comes {nxt}"
    return dict(canon=canon, q=canon, a=str(nxt), expr=None, work=work)


GENERATORS = [(gen_add, 0.25), (gen_sub, 0.15), (gen_mul, 0.20),
              (gen_compare, 0.20), (gen_sequence, 0.20)]


def render(p):
    r = random.random()
    if r < 0.3 and p["work"]:
        return f"Question: {p['q']}\nWork: {p['work']}\nAnswer: {p['a']}"
    if r < 0.5 and p["expr"]:
        return f"{p['expr']} = {p['a']}"
    return f"Question: {p['q']}\nAnswer: {p['a']}"


def main():
    with open(EVAL_PATH, encoding="utf-8") as f:
        eval_questions = {json.loads(line)["question"] for line in f}

    funcs = [g for g, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written, skipped, samples = 0, 0, []
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        while written < N_EXAMPLES:
            p = random.choices(funcs, weights)[0]()
            if p["canon"] in eval_questions:
                skipped += 1
                continue
            text = render(p)
            if len(samples) < 8:
                samples.append(text)
            f.write(json.dumps({"text": text}) + "\n")
            written += 1

    print(f"wrote {written} examples to {OUT_PATH}, skipped {skipped} that matched the eval set\n")
    for s in samples:
        print(s)
        print("---")


if __name__ == "__main__":
    main()