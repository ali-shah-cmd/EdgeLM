import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "eval_data" / "reasoning_eval.jsonl"
OLD_EVAL_PATH = ROOT / "eval_data" / "math_eval.jsonl"

SEED = 20261119
N_PER_ARITHMETIC_SUBTYPE = 10  # 6 subtypes -> 60 arithmetic questions
N_PER_WORD_SUBTYPE = 12        # 5 subtypes -> 60 word-problem questions
N_PER_LOGIC_SUBTYPE = 20       # 3 subtypes -> 60 logic questions


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_existing_questions():
    used = set()
    if OLD_EVAL_PATH.exists():
        with OLD_EVAL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    q = row.get("question")
                    if q:
                        used.add(norm(q))
                except json.JSONDecodeError:
                    continue
    return used


def build_examples():
    rng = random.Random(SEED)
    rows = []
    used = load_existing_questions()

    def add(category, subcategory, question, answer, style):
        key = norm(question)
        if key in used:
            raise RuntimeError(f"Duplicate evaluation question detected: {question}")
        used.add(key)

        if style == "direct":
            prompt = f"Question: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nWork:"

        rows.append({
            "category": category,
            "subcategory": subcategory,
            "question": question,
            "answer": str(answer),
            "prompt": prompt,
        })

    # ---------------------------------------------------------
    # Arithmetic: direct calculations, comparisons, sequences.
    # Wording intentionally differs from the training examples.
    # ---------------------------------------------------------
    for _ in range(N_PER_ARITHMETIC_SUBTYPE):
        a, b = rng.randint(12, 999), rng.randint(11, 899)
        add("arithmetic", "addition",
            f"Calculate the total when {a} is increased by {b}. Give only the integer.",
            a + b, "direct")

        a = rng.randint(100, 999)
        b = rng.randint(10, a - 1)
        add("arithmetic", "subtraction",
            f"Find the result of taking {b} away from {a}. Give only the integer.",
            a - b, "direct")

        a, b = rng.randint(3, 39), rng.randint(3, 29)
        add("arithmetic", "multiplication",
            f"Determine the product obtained from {a} multiplied by {b}. Give only the integer.",
            a * b, "direct")

        divisor = rng.randint(2, 15)
        quotient = rng.randint(8, 80)
        dividend = divisor * quotient
        add("arithmetic", "division",
            f"Work out {dividend} divided equally by {divisor}. Give only the integer.",
            quotient, "direct")

        a, b = rng.sample(range(20, 1000), 2)
        add("arithmetic", "comparison",
            f"Which is the greater value, {a} or {b}? Respond with the greater number only.",
            max(a, b), "direct")

        start = rng.randint(2, 50)
        step = rng.randint(2, 17)
        seq = [start + i * step for i in range(4)]
        add("arithmetic", "sequence",
            f"Continue the pattern by one value: {', '.join(map(str, seq))}. What value comes next? Give only the number.",
            start + 4 * step, "direct")

    # ---------------------------------------------------------
    # Word problems: alternate scenarios and sentence patterns.
    # ---------------------------------------------------------
    items = ["notebooks", "marbles", "folders", "bottles", "tickets",
             "oranges", "cards", "pencils", "books", "stickers"]
    people = ["Mina", "Rafi", "Sara", "Omar", "Lena", "Noah", "Ira", "Zara"]

    for _ in range(N_PER_WORD_SUBTYPE):
        a, b = rng.randint(35, 450), rng.randint(20, 350)
        item, person = rng.choice(items), rng.choice(people)
        add("word_problems", "addition",
            f"{person} counted {a} {item} in the morning and found {b} more later. How many {item} did {person} count altogether?",
            a + b, "work")

        a = rng.randint(150, 900)
        b = rng.randint(15, min(140, a - 1))
        item = rng.choice(items)
        add("word_problems", "subtraction",
            f"A storeroom began with {a} {item}. After {b} were removed, how many {item} remained?",
            a - b, "work")

        packs, per_pack = rng.randint(4, 24), rng.randint(5, 35)
        item = rng.choice(items)
        add("word_problems", "multiplication",
            f"A delivery contains {packs} identical bundles, each holding {per_pack} {item}. How many {item} are in the delivery?",
            packs * per_pack, "work")

        divisor, quotient = rng.randint(3, 16), rng.randint(8, 65)
        total = divisor * quotient
        item = rng.choice(items)
        add("word_problems", "division",
            f"A coordinator shares {total} {item} equally among {divisor} teams. How many {item} does each team receive?",
            quotient, "work")

        packs, per_pack = rng.randint(5, 18), rng.randint(6, 24)
        total = packs * per_pack
        removed = rng.randint(1, total - 1)
        item = rng.choice(items)
        add("word_problems", "two_step",
            f"A shop receives {packs} cartons containing {per_pack} {item} apiece. It then sells {removed} {item}. How many are still in the shop?",
            total - removed, "work")

    # ---------------------------------------------------------
    # Logic: universal rules, conditional inference, ordering.
    # ---------------------------------------------------------
    names = ["Tariq", "Maya", "Ibrahim", "Nora", "Haris", "Lina",
             "Dani", "Amira", "Bilal", "Sana", "Rayan", "Huda",
             "Sameer", "Aisha", "Kamil", "Noor", "Yusuf", "Mariam",
             "Zain", "Farah"]
    groups = ["choir", "science", "drama", "chess", "robotics",
              "gardening", "debate", "music", "art", "coding"]
    properties = ["a blue badge", "a library card", "a safety pass",
                  "a team uniform", "a practice key"]

    for i in range(N_PER_LOGIC_SUBTYPE):
        person = names[i % len(names)]
        group = groups[i % len(groups)]
        prop = properties[i % len(properties)]
        if i % 2 == 0:
            q = (f"The club rule states that every {group} club member must have {prop}. "
                 f"{person} is a {group} club member. Must {person} have {prop}? Answer yes or no.")
            ans = "yes"
        else:
            q = (f"Everyone in the {group} club has {prop}. "
                 f"{person} does not have {prop}. Can {person} be a member of that club? Answer yes or no.")
            ans = "no"
        add("logic", "deduction", q, ans, "work")

    for i in range(N_PER_LOGIC_SUBTYPE):
        actor = names[(i + 2) % len(names)]
        actions = [
            ("If the sensor detects smoke, the warning light turns on.",
             "The sensor detects smoke.",
             "Can we infer that the warning light turns on?", "yes"),
            ("If the gate is locked, the alarm is enabled.",
             "The alarm is enabled.",
             "Can we conclude that the gate is locked?", "no"),
            ("If the server loses power, the status screen goes dark.",
             "The status screen is not dark.",
             "Can we conclude that the server did not lose power?", "yes"),
            ("If a student submits the form, the office sends a receipt.",
             "The student did not submit the form.",
             "Can we conclude that the office did not send a receipt?", "no"),
            ("If the temperature falls below zero, the pond may freeze.",
             "The temperature falls below zero.",
             "Does the rule guarantee that the pond freezes?", "no"),
        ]
        rule, fact, question, ans = actions[i % len(actions)]
        q = f"{rule} {actor} is described as follows: {fact} {question} Answer yes or no."
        add("logic", "conditional", q, ans, "work")

    for _ in range(N_PER_LOGIC_SUBTYPE):
        trio = rng.sample(["Ari", "Bea", "Cem", "Devi", "Evan",
                           "Faye", "Gus", "Hana", "Ivan", "Jude"], 3)
        low, mid, high = trio
        if rng.choice([True, False]):
            facts = f"{low} is shorter than {mid}, and {mid} is shorter than {high}."
            question = "Who is the tallest?"
            answer = high
        else:
            facts = f"{low} is faster than {mid}, and {mid} is faster than {high}."
            question = "Who is the slowest?"
            answer = high
        q = f"Consider these comparisons: {facts} {question}"
        add("logic", "ordering", q, answer, "work")

    rng.shuffle(rows)
    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = build_examples()

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = {}
    for row in rows:
        key = (row["category"], row["subcategory"])
        counts[key] = counts.get(key, 0) + 1

    print(f"Saved {len(rows):,} held-out questions to: {OUT_PATH}")
    print("\nDistribution:")
    for (category, subcategory), count in sorted(counts.items()):
        print(f"  {category:14s} | {subcategory:14s} | {count:3d}")
    print("\nThese questions are fixed by seed. Do not regenerate between baseline and post-training evaluation.")


if __name__ == "__main__":
    main()
