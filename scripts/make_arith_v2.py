import argparse
import json
import random
import re
from pathlib import Path

# Arithmetic v2: many different question wordings + step-by-step "Work:" lines.
# Run from the project root (C:\Work\SLM). Does not touch any existing file.

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=500_000)
parser.add_argument("--seed", type=int, default=20261120)
parser.add_argument("--out", default="data/raw/arith_v2.jsonl")
parser.add_argument("--eval", default="eval_data/reasoning_eval.jsonl")
parser.add_argument("--direct_pct", type=int, default=15,
                    help="Percent of examples WITHOUT a Work line (answer only).")
args = parser.parse_args()

rng = random.Random(args.seed)


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


# Questions in the frozen eval must never appear in training data.
eval_questions = set()
if Path(args.eval).exists():
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_questions.add(norm(json.loads(line)["question"]))
print(f"Loaded {len(eval_questions)} eval questions to exclude.")


# ---------------------------------------------------------------
# Question wordings. None of these is a copy of an eval template.
# ---------------------------------------------------------------
ADD = [
    "What is {a} plus {b}?", "Find the sum of {a} and {b}.", "{a} + {b} = ?",
    "Add {b} to {a}.", "What is the total of {a} and {b}?", "Compute {a} + {b}.",
    "Add {a} and {b}.", "Work out the sum: {a} + {b}.", "What do you get by adding {a} and {b}?",
    "Sum {a} and {b}.", "How much is {a} + {b}?",
]
SUB = [
    "What is {a} minus {b}?", "Subtract {b} from {a}.", "{a} - {b} = ?",
    "What is {a} - {b}?", "Compute {a} - {b}.", "Start with {a} and remove {b}. What is left?",
    "How much is {a} take away {b}?", "Take {b} from {a}.", "Calculate {a} minus {b}.",
]
MUL = [
    "What is {a} times {b}?", "Multiply {a} by {b}.", "{a} * {b} = ?",
    "Find the product of {a} and {b}.", "Calculate {a} x {b}.", "Compute {a} * {b}.",
    "What do you get when you multiply {a} and {b}?", "How much is {a} times {b}?",
]
DIV = [
    "What is {a} divided by {b}?", "Divide {a} by {b}.", "{a} / {b} = ?",
    "How many times does {b} go into {a}?", "Split {a} into {b} equal parts. How big is each part?",
    "Find the quotient of {a} and {b}.", "Compute {a} / {b}.", "Calculate {a} / {b}.",
]
CMP_MAX = [
    "Which number is larger, {a} or {b}?", "Is {a} or {b} bigger?",
    "Pick the bigger number: {a} or {b}.", "What is the maximum of {a} and {b}?",
    "Between {a} and {b}, which is greater?",
]
CMP_MIN = [
    "Which number is smaller, {a} or {b}?", "Is {a} or {b} less?",
    "Pick the smaller number: {a} or {b}.", "What is the minimum of {a} and {b}?",
    "Between {a} and {b}, which is the lowest?",
]
SEQ = [
    "What comes next: {s}, ?", "What is the next number in {s}?",
    "Find the next term: {s}, ?", "Continue the sequence: {s}, ...",
    "{s}, ... what number follows?",
]
SUFFIXES = ["", "", "", " Answer with a number only.", " Reply with just the number."]


# ---------------------------------------------------------------
# Helpers: split a number into place values (347 -> [300, 40, 7])
# ---------------------------------------------------------------
def parts(n):
    s = str(n)
    return [int(ch) * 10 ** (len(s) - 1 - i) for i, ch in enumerate(s)]


def nonzero_parts(n):
    return [p for p in parts(n) if p > 0]


def rand_num(lo=2):
    r = rng.random()
    if r < 0.25:
        return rng.randint(lo, 99)
    if r < 0.80:
        return rng.randint(100, 999)
    return rng.randint(1000, 9999)


# ---------------------------------------------------------------
# Work-line writers. Each ends with "= <answer>".
# ---------------------------------------------------------------
def work_add(a, b):
    pa, pb = parts(a), parts(b)
    width = max(len(pa), len(pb))
    pa = [0] * (width - len(pa)) + pa
    pb = [0] * (width - len(pb)) + pb
    if width == 1:
        return f"{a} + {b} = {a + b}"
    pairs = " + ".join(f"({x} + {y})" for x, y in zip(pa, pb))
    sums = " + ".join(str(x + y) for x, y in zip(pa, pb))
    return f"{a} + {b} = {pairs} = {sums} = {a + b}"


def work_sub(a, b):
    ps = nonzero_parts(b)
    if len(ps) == 1:
        return f"{a} - {b} = {a - b}"
    chain = [f"{a} - " + " - ".join(map(str, ps))]
    cur = a
    for i, p in enumerate(ps[:-1]):
        cur -= p
        chain.append(f"{cur} - " + " - ".join(map(str, ps[i + 1:])))
    chain.append(str(a - b))
    return f"{a} - {b} = " + " = ".join(chain)


def work_mul(a, b):
    pas, pbs = nonzero_parts(a), nonzero_parts(b)
    if len(pas) == 1 and len(pbs) == 1:
        return f"{a} * {b} = {a * b}"
    terms = [(x, y) for x in pas for y in pbs]
    text = f"{a} * {b} = " + " + ".join(f"{x} * {y}" for x, y in terms)
    text += " = " + " + ".join(str(x * y) for x, y in terms)
    if len(pas) > 1 and len(pbs) > 1:
        groups = [sum(x * y for y in pbs) for x in pas]
        text += " = " + " + ".join(map(str, groups))
    return text + f" = {a * b}"


def work_div(a, b):
    q = a // b
    ps = nonzero_parts(q)
    if len(ps) == 1:
        return f"{b} * {q} = {a}, so {a} / {b} = {q}"
    rem = a
    steps = []
    for p in ps:
        prod = b * p
        steps.append(f"{b} * {p} = {prod}, {rem} - {prod} = {rem - prod}")
        rem -= prod
    return f"{a} / {b}: " + "; ".join(steps) + f"; {' + '.join(map(str, ps))} = {q}"


# ---------------------------------------------------------------
# Example makers: return (question, work, answer, category)
# ---------------------------------------------------------------
def make_add():
    a, b = rand_num(), rand_num()
    return rng.choice(ADD).format(a=a, b=b), work_add(a, b), a + b, "add"


def make_sub():
    a = rand_num(12)
    b = rng.randint(1, a - 1)
    return rng.choice(SUB).format(a=a, b=b), work_sub(a, b), a - b, "sub"


def make_mul():
    a = rng.randint(2, 99)
    b = rng.randint(2, 9) if rng.random() < 0.5 else rng.randint(2, 99)
    if rng.random() < 0.5:
        a, b = b, a
    return rng.choice(MUL).format(a=a, b=b), work_mul(a, b), a * b, "mul"


def make_div():
    b = rng.randint(2, 20)
    q = rng.randint(2, 150) if rng.random() < 0.8 else rng.randint(150, 999)
    a = b * q
    return rng.choice(DIV).format(a=a, b=b), work_div(a, b), q, "div"


def make_cmp():
    a, b = rng.sample(range(2, 10000), 2)
    if rng.random() < 0.5:
        a, b = rng.sample(range(2, 1000), 2)
    if rng.random() < 0.5:
        q, ans = rng.choice(CMP_MAX).format(a=a, b=b), max(a, b)
    else:
        q, ans = rng.choice(CMP_MIN).format(a=a, b=b), min(a, b)
    sign = ">" if a > b else "<"
    return q, f"{a} {sign} {b}", ans, "cmp"


def make_seq():
    start = rng.randint(2, 200)
    step = rng.randint(2, 25)
    if rng.random() < 0.25:  # decreasing sequence
        start += 4 * step
        step = -step
    seq = [start + i * step for i in range(4)]
    nxt = start + 4 * step
    s = ", ".join(map(str, seq))
    sign = "+" if step > 0 else "-"
    work = f"{s}: step is {step}; {seq[-1]} {sign} {abs(step)} = {nxt}"
    return rng.choice(SEQ).format(s=s), work, nxt, "seq"


MAKERS = [(make_add, 0.20), (make_sub, 0.20), (make_mul, 0.20),
          (make_div, 0.15), (make_cmp, 0.10), (make_seq, 0.15)]
funcs = [f for f, _ in MAKERS]
weights = [w for _, w in MAKERS]

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

written = skipped = 0
samples = []
with out_path.open("w", encoding="utf-8") as f:
    while written < args.n:
        question, work, answer, kind = rng.choices(funcs, weights=weights, k=1)[0]()
        question += rng.choice(SUFFIXES)

        # Safety check: a Work line must end in the right answer.
        if kind in ("add", "sub", "mul", "div"):
            assert work.rstrip().endswith(f"= {answer}"), (question, work, answer)

        if norm(question) in eval_questions:
            skipped += 1
            continue

        if rng.randint(1, 100) <= args.direct_pct:
            text = f"Question: {question}\nAnswer: {answer}"
        else:
            text = f"Question: {question}\nWork: {work}\nAnswer: {answer}"

        record = {"text": text, "category": f"arith_v2_{kind}",
                  "question": question, "answer": str(answer)}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(samples) < 12:
            samples.append(text)
        written += 1

print(f"Wrote {written:,} examples to {out_path}; skipped {skipped} eval-question matches.\n")
for s in samples:
    print(s)
    print("---")