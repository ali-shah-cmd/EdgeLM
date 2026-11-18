import json
import random
from pathlib import Path

# Step 2b: generate word-problem and logic data.
# The existing tokenizer and model are not modified.
SEED = 20261117
N_WORD = 1_000_000
N_LOGIC = 1_000_000

EVAL_PATH = Path("eval_data/math_eval.jsonl")
WORD_OUT = Path("data/raw/word_problems.jsonl")
LOGIC_OUT = Path("data/raw/logic_synth.jsonl")

rng = random.Random(SEED)

NAMES = ["Ava", "Ben", "Mia", "Omar", "Zara", "Leo", "Sara", "Noah",
         "Iris", "Adam", "Lina", "Ravi", "Nora", "Eli", "Maya", "Hana"]
ITEMS = ["apples", "books", "pencils", "oranges", "bottles", "stickers",
         "notebooks", "marbles", "tickets", "cups", "packets", "cards"]
PLACES = ["a shop", "a school", "a library", "a garden", "a warehouse",
          "a bakery", "a sports center"]
UNITS = ["kilometers", "meters", "minutes", "rupees", " liters", "pages"]
ANIMALS = ["foxes", "whales", "eagles", "rabbits", "turtles", "dolphins"]
GROUPS = ["musicians", "athletes", "artists", "students", "gardeners", "pilots"]
PROPERTIES = ["a badge", "a notebook", "a special pass",
              "a blue uniform", "a place on the morning team"]

ANIMAL_SINGULAR = {
    "foxes": "fox", "whales": "whale", "eagles": "eagle",
    "rabbits": "rabbit", "turtles": "turtle", "dolphins": "dolphin",
}


def load_eval_questions():
    questions = set()
    if EVAL_PATH.exists():
        with EVAL_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if "question" in row:
                        questions.add(row["question"].strip())
                except (json.JSONDecodeError, TypeError):
                    continue
    # Also avoid the exact few-shot questions used by math_eval.py.
    questions.update({
        "What is 3 + 4?",
        "What is 14 * 6?",
        "Which number is larger, 61 or 38?",
        "What comes next: 5, 8, 11, 14, ?",
        "Ben has 8 cups. Ben buys 5 more. How many cups does Ben have now?",
        "All hens are birds. Sue is a hen. Is Sue a bird? Answer yes or no.",
        "No cows are fish. Sue is a cow. Is Sue a fish? Answer yes or no.",
    })
    return questions


def make_record(category, question, work, answer):
    text = f"Question: {question}\nWork: {work}\nAnswer: {answer}"
    return {
        "text": text,
        "category": category,
        "question": question,
        "answer": str(answer),
    }


# -------------------- Word problems --------------------

def word_add():
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    a, b = rng.randint(12, 450), rng.randint(12, 450)
    q = (f"{name} has {a} {item}. Later, {name} gets {b} more "
         f"{item}. How many {item} does {name} have now?")
    return make_record("word_add", q, f"{a} + {b} = {a+b}", a+b)


def word_sub():
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    a = rng.randint(30, 900)
    b = rng.randint(1, a)
    q = (f"A store has {a} {item}. It sells {b} of them. "
         f"How many {item} remain?")
    return make_record("word_sub", q, f"{a} - {b} = {a-b}", a-b)


def word_mul():
    item = rng.choice(ITEMS)
    groups = rng.randint(2, 30)
    each = rng.randint(2, 40)
    q = (f"There are {groups} boxes with {each} {item} in each box. "
         f"How many {item} are there altogether?")
    ans = groups * each
    return make_record("word_mul", q, f"{groups} * {each} = {ans}", ans)


def word_div():
    item = rng.choice(ITEMS)
    people = rng.randint(2, 20)
    each = rng.randint(2, 50)
    total = people * each
    q = (f"{total} {item} are shared equally among {people} people. "
         f"How many {item} does each person get?")
    return make_record("word_div", q, f"{total} / {people} = {each}", each)


def word_two_step():
    name = rng.choice(NAMES)
    item = rng.choice(ITEMS)
    packs = rng.randint(2, 20)
    each = rng.randint(3, 30)
    used = rng.randint(1, packs * each - 1)
    total = packs * each
    ans = total - used
    q = (f"{name} buys {packs} packs of {item}, with {each} {item} "
         f"in each pack. {name} then gives away {used} {item}. "
         f"How many {item} are left?")
    work = f"{packs} * {each} = {total}; {total} - {used} = {ans}"
    return make_record("word_two_step", q, work, ans)


def word_distance_or_money():
    name = rng.choice(NAMES)
    unit = rng.choice(["rupees", "kilometers", "meters", "minutes"])
    a, b = rng.randint(10, 400), rng.randint(10, 400)
    q = (f"{name} records {a} {unit} on the first day and "
         f"{b} {unit} on the second day. What is the total?")
    return make_record("word_add", q, f"{a} + {b} = {a+b}", a+b)


WORD_GENERATORS = [
    (word_add, 0.22),
    (word_sub, 0.18),
    (word_mul, 0.20),
    (word_div, 0.18),
    (word_two_step, 0.17),
    (word_distance_or_money, 0.05),
]


# -------------------- Logic problems --------------------

def logic_order():
    people = rng.sample(NAMES, 3)
    ranked = rng.sample([1, 2, 3], 3)  # 1 = shortest, 3 = tallest
    heights = dict(zip(people, ranked))
    # State two adjacent comparisons, then ask for the extreme.
    ordered = sorted(people, key=lambda n: heights[n])
    if rng.random() < 0.5:
        q = (f"{ordered[2]} is taller than {ordered[1]}. "
             f"{ordered[1]} is taller than {ordered[0]}. "
             f"Who is the shortest?")
        ans = ordered[0]
        work = f"{ordered[2]} > {ordered[1]} > {ordered[0]} in height"
    else:
        q = (f"{ordered[0]} is shorter than {ordered[1]}. "
             f"{ordered[1]} is shorter than {ordered[2]}. "
             f"Who is the tallest?")
        ans = ordered[2]
        work = f"{ordered[0]} < {ordered[1]} < {ordered[2]} in height"
    return make_record("logic_order", q, work, ans)


def logic_all_yes():
    group = rng.choice(GROUPS)
    prop = rng.choice(PROPERTIES)
    name = rng.choice(NAMES)
    q = (f"Every member of the {group} group has {prop}. "
         f"{name} is a member of the {group} group. "
         f"Does {name} have {prop}? Answer yes or no.")
    return make_record("logic_deduction", q, "The rule applies to every group member.", "yes")


def logic_all_no():
    group = rng.choice(GROUPS)
    prop = rng.choice(PROPERTIES)
    name = rng.choice(NAMES)
    q = (f"Every member of the {group} group has {prop}. "
         f"{name} does not have {prop}. "
         f"Can we conclude that {name} is not a member of the {group} group? "
         f"Answer yes or no.")
    # If every member has the property, a person lacking it cannot be a member.
    return make_record("logic_deduction", q, "A group member must have the stated property.", "yes")


def logic_exclusion():
    a, b = rng.sample(ANIMALS, 2)
    name = rng.choice(NAMES)
    a_singular, b_singular = ANIMAL_SINGULAR[a], ANIMAL_SINGULAR[b]
    q = (f"No {a} are {b}. {name} is a {a_singular}. "
         f"Is {name} a {b_singular}? Answer yes or no.")
    return make_record("logic_deduction", q, f"No {a} belong to the {b} group.", "no")


def logic_conditional_yes():
    name = rng.choice(NAMES)
    action, negative_action = rng.choice([
        ("studies", "does not study"),
        ("waters the plants", "does not water the plants"),
        ("sets an alarm", "does not set an alarm"),
        ("charges a device", "does not charge a device"),
        ("locks the door", "does not lock the door"),
    ])
    result = rng.choice(["is prepared", "the plants receive water",
                         "the alarm is active", "the device has power",
                         "the door is secured"])
    q = (f"If {name} {action}, then {result}. "
         f"{name} {action}. Can we conclude that {result}? Answer yes or no.")
    return make_record("logic_conditional", q, "The condition is true, so its stated result follows.", "yes")


def logic_conditional_unknown():
    name = rng.choice(NAMES)
    action, negative_action = rng.choice([
        ("studies", "does not study"),
        ("sets an alarm", "does not set an alarm"),
        ("charges a device", "does not charge a device"),
        ("locks the door", "does not lock the door"),
        ("takes the bus", "does not take the bus"),
    ])
    result = rng.choice(["passes the test", "the alarm is active",
                         "the device has power", "the door is secured",
                         "arrives at school"])
    q = (f"If {name} {action}, then {result}. "
         f"We know that {name} {negative_action}. "
         f"Can we conclude that {result} is false? Answer yes or no.")
    # Not doing the antecedent does not establish that the result is false.
    return make_record("logic_conditional", q, "The rule does not say what happens when the condition is false.", "no")


def logic_generality():
    # A simple valid syllogism with deliberately varied vocabulary.
    animal = rng.choice(ANIMALS)
    name = rng.choice(NAMES)
    singular = ANIMAL_SINGULAR[animal]
    q = (f"All {animal} are living things. {name} is a {singular}. "
         f"Is {name} a living thing? Answer yes or no.")
    return make_record("logic_deduction", q, "The second statement places the person in the stated category.", "yes")


LOGIC_GENERATORS = [
    (logic_order, 0.20),
    (logic_all_yes, 0.20),
    (logic_all_no, 0.15),
    (logic_exclusion, 0.15),
    (logic_conditional_yes, 0.15),
    (logic_conditional_unknown, 0.10),
    (logic_generality, 0.05),
]


def write_dataset(path, count, generators, excluded, label):
    funcs = [fn for fn, _ in generators]
    weights = [weight for _, weight in generators]
    written = skipped = 0
    samples = []
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        while written < count:
            record = rng.choices(funcs, weights=weights, k=1)[0]()
            if record["question"].strip() in excluded:
                skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(samples) < 5:
                samples.append(record)
            written += 1

    print(f"{label}: wrote {written:,} examples to {path}; skipped {skipped} exact eval-question matches")
    for row in samples:
        print(json.dumps(row, ensure_ascii=False))
    print()


def main():
    excluded = load_eval_questions()
    print(f"Loaded {len(excluded)} evaluation/few-shot questions to exclude.")
    write_dataset(WORD_OUT, N_WORD, WORD_GENERATORS, excluded, "Word problems")
    write_dataset(LOGIC_OUT, N_LOGIC, LOGIC_GENERATORS, excluded, "Logic")


if __name__ == "__main__":
    main()
