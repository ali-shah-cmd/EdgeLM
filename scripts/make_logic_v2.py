import argparse
import json
import random
import re
from pathlib import Path

# Logic v2: many wordings, balanced yes/no, and a Work line that really uses
# the facts of each question (not one canned sentence per template).
# Run from the project root (C:\Work\SLM). Does not touch any existing file.

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=500_000)
parser.add_argument("--seed", type=int, default=20261121)
parser.add_argument("--out", default="data/raw/logic_v2.jsonl")
parser.add_argument("--eval", default="eval_data/reasoning_eval.jsonl")
args = parser.parse_args()

rng = random.Random(args.seed)


def norm(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def cap(s):
    return s[0].upper() + s[1:]


def art(word):
    return "an" if word[0].lower() in "aeiou" else "a"


eval_questions = set()
if Path(args.eval).exists():
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                eval_questions.add(norm(json.loads(line)["question"]))
print(f"Loaded {len(eval_questions)} eval questions to exclude.")


# ---------------------------------------------------------------
# Vocabulary pools (large on purpose, so questions cannot be memorized)
# ---------------------------------------------------------------
NAMES = """Aaron Abby Adel Aiden Alma Amir Anya Arlo Asha Axel Bram Cleo Cyrus Dara Dev Edna Elio Emre Enzo
Esme Faris Fay Finn Gita Hugo Idris Ila Inez Jago Jana Joss Kai Kira Lars Leila Liam Lola Luca Mara Milo
Mina Nadia Nico Nia Odin Olga Omar Pia Quin Rami Reza Rhea Rory Ruth Sami Sasha Selim Tala Teo Thea Timo
Uma Vera Vik Wren Xena Yara Yuki Zane Zeke Zoya Abdul Bella Carlos Diana Elena Felix Grace Hassan Isla
Jonas Karim Layla Marco Nina Oscar Petra Rosa Stefan Talia Ulrich Vince Willa Yasmin Zahra Ayaan Bushra
Daniyal Eesha Fahad Gulnar Hira Iqbal Jawad Kiran Laiba Moiz Noman Osama Parveen Qasim Rida Saad Tuba""".split()

GROUPS = [
    ("pilots", "pilot"), ("bakers", "baker"), ("nurses", "nurse"), ("farmers", "farmer"),
    ("drivers", "driver"), ("tutors", "tutor"), ("painters", "painter"), ("sailors", "sailor"),
    ("welders", "welder"), ("chefs", "chef"), ("librarians", "librarian"), ("dancers", "dancer"),
    ("mechanics", "mechanic"), ("lifeguards", "lifeguard"), ("carpenters", "carpenter"),
    ("guides", "guide"), ("referees", "referee"), ("pharmacists", "pharmacist"),
    ("electricians", "electrician"), ("coaches", "coach"), ("plumbers", "plumber"),
    ("surveyors", "surveyor"), ("potters", "potter"), ("editors", "editor"),
    ("gardeners", "gardener"), ("tailors", "tailor"), ("rangers", "ranger"), ("clerks", "clerk"),
]

PROPS = [
    "a red badge", "a parking permit", "a toolbox", "a library card", "a night pass",
    "a safety helmet", "a locker key", "a city map", "a blue folder", "a signed form",
    "a training certificate", "a radio", "a gate code", "a lunch ticket", "a work phone",
    "a steel bottle", "an ID card", "a warm jacket", "a season ticket", "a visitor sticker",
    "a first-aid kit", "a name tag", "a spare key", "an entry stamp",
]

# (A true, A false, B true, B false). "{p}" is replaced by a person's name.
COND = [
    ("the oven is hot", "the oven is not hot", "the bread bakes quickly", "the bread does not bake quickly"),
    ("the battery is charged", "the battery is not charged", "the lamp turns on", "the lamp does not turn on"),
    ("it rains overnight", "it does not rain overnight", "the road is wet", "the road is not wet"),
    ("the bridge is closed", "the bridge is not closed", "traffic takes the highway", "traffic does not take the highway"),
    ("the printer has paper", "the printer has no paper", "the report gets printed", "the report does not get printed"),
    ("the teacher is absent", "the teacher is not absent", "the class ends early", "the class does not end early"),
    ("the river floods", "the river does not flood", "the farm loses its crops", "the farm does not lose its crops"),
    ("the bell rings", "the bell does not ring", "the students line up", "the students do not line up"),
    ("the shop is open", "the shop is not open", "the lights are on", "the lights are not on"),
    ("the freezer breaks", "the freezer does not break", "the ice cream melts", "the ice cream does not melt"),
    ("the team wins the match", "the team does not win the match", "the crowd cheers", "the crowd does not cheer"),
    ("the tank is full", "the tank is not full", "the car drives to the coast", "the car does not drive to the coast"),
    ("the window is open", "the window is not open", "the room gets cold", "the room does not get cold"),
    ("the plant gets sunlight", "the plant gets no sunlight", "the plant grows", "the plant does not grow"),
    ("the ticket is valid", "the ticket is not valid", "the turnstile lets the visitor in", "the turnstile does not let the visitor in"),
    ("the password is correct", "the password is not correct", "the account unlocks", "the account does not unlock"),
    ("the soup boils", "the soup does not boil", "the pot lid rattles", "the pot lid does not rattle"),
    ("the wind is strong", "the wind is not strong", "the kite flies high", "the kite does not fly high"),
    ("the alarm clock rings", "the alarm clock does not ring", "the baker wakes up", "the baker does not wake up"),
    ("the road is icy", "the road is not icy", "the bus runs late", "the bus does not run late"),
    ("the lake freezes", "the lake does not freeze", "the skaters come out", "the skaters do not come out"),
    ("the cat is hungry", "the cat is not hungry", "the cat meows loudly", "the cat does not meow loudly"),
    ("the delivery arrives", "the delivery does not arrive", "the shelf gets restocked", "the shelf does not get restocked"),
    ("the signal is red", "the signal is not red", "the cars stop", "the cars do not stop"),
    ("the pump works", "the pump does not work", "the well fills up", "the well does not fill up"),
    ("the cake is ready", "the cake is not ready", "the guests sing", "the guests do not sing"),
    ("the camera is on", "the camera is not on", "the screen shows a picture", "the screen does not show a picture"),
    ("the match is lit", "the match is not lit", "the candle burns", "the candle does not burn"),
    ("the fog is thick", "the fog is not thick", "the flights are delayed", "the flights are not delayed"),
    ("{p} trains daily", "{p} does not train daily", "{p} gets stronger", "{p} does not get stronger"),
    ("{p} reads the notes", "{p} does not read the notes", "{p} passes the quiz", "{p} does not pass the quiz"),
    ("{p} leaves early", "{p} does not leave early", "{p} catches the train", "{p} does not catch the train"),
    ("{p} waters the herbs", "{p} does not water the herbs", "the herbs stay green", "the herbs do not stay green"),
]

# (comparative for more, comparative for less, superlative max, superlative min)
ADJ = [
    ("taller", "shorter", "tallest", "shortest"),
    ("faster", "slower", "fastest", "slowest"),
    ("older", "younger", "oldest", "youngest"),
    ("heavier", "lighter", "heaviest", "lightest"),
    ("stronger", "weaker", "strongest", "weakest"),
    ("richer", "poorer", "richest", "poorest"),
    ("louder", "quieter", "loudest", "quietest"),
]

YN = ["Answer yes or no.", "Reply with yes or no."]
QFR = ["Can we conclude that {X}?", "Does it follow that {X}?",
       "Is it certain that {X}?", "Must it be true that {X}?"]


# ---------------------------------------------------------------
# Conditional reasoning (if A then B)
# ---------------------------------------------------------------
def make_conditional():
    A, nA, B, nB = rng.choice(COND)
    p = rng.choice(NAMES)
    A, nA, B, nB = [s.replace("{p}", p) for s in (A, nA, B, nB)]
    rule = rng.choice(["If {A}, then {B}.", "If {A}, {B}.", "Whenever {A}, {B}."]).format(A=A, B=B)

    if rng.random() < 0.5:
        form = rng.choice(["mp", "mt"])
    else:
        form = rng.choice(["da", "ac", "mpneg", "mtneg"])

    head = f"Rule: {A} -> {B}."
    if form == "mp":
        fact, concl, ans = A, B, "yes"
        work = f"{head} Fact: {A}. So {B}."
    elif form == "mt":
        fact, concl, ans = nB, nA, "yes"
        work = f"{head} Fact: {nB}. If {A}, then {B}, so {nA}."
    elif form == "da":
        fact, concl, ans = nA, nB, "no"
        work = f"{head} Fact: {nA}. The rule does not say what happens otherwise, so nothing follows."
    elif form == "ac":
        fact, concl, ans = B, A, "no"
        work = f"{head} Fact: {B}. That could happen for other reasons, so {A} is not certain."
    elif form == "mpneg":
        fact, concl, ans = A, nB, "no"
        work = f"{head} Fact: {A}. So {B}. The claim says the opposite, so it is false."
    else:  # mtneg
        fact, concl, ans = nB, A, "no"
        work = f"{head} Fact: {nB}. So {nA}. The claim says the opposite, so it is false."

    question = f"{rule} {cap(fact)}. {rng.choice(QFR).format(X=concl)} {rng.choice(YN)}"
    return question, work, ans, "conditional_" + form


# ---------------------------------------------------------------
# Universal / exclusion / "some" reasoning about groups
# ---------------------------------------------------------------
def make_universal():
    pl, sg = rng.choice(GROUPS)
    P = rng.choice(PROPS)
    X = rng.choice(NAMES)
    a_sg = f"{art(sg)} {sg}"

    pos = rng.choice([f"All {pl} have {P}.", f"Every {sg} has {P}.", f"Each {sg} has {P}."])
    neg = rng.choice([f"No {pl} have {P}.", f"None of the {pl} have {P}."])
    some = rng.choice([f"Some {pl} have {P}.", f"Only some of the {pl} have {P}."])
    is_g, not_g = f"{X} is {a_sg}.", f"{X} is not {a_sg}."
    has_p, lacks_p = f"{X} has {P}.", f"{X} does not have {P}."

    form = rng.choice(["u1", "u1b", "u2a", "u2b", "u2c", "u3", "u4", "u4b", "u5", "u6"])
    if form == "u1":
        rule, fact, q, ans = pos, is_g, f"Does {X} have {P}?", "yes"
        work = f"Rule: all {pl} have {P}. {X} is {a_sg}. So {X} has {P}."
    elif form == "u1b":
        rule, fact, q, ans = pos, is_g, f"Can we conclude that {X} has {P}?", "yes"
        work = f"Rule: all {pl} have {P}. {X} is {a_sg}. So {X} has {P}."
    elif form == "u2a":
        rule, fact, q, ans = pos, lacks_p, f"Is {X} {a_sg}?", "no"
        work = f"Rule: all {pl} have {P}. {X} does not have {P}. So {X} is not {a_sg}."
    elif form == "u2b":
        rule, fact, q, ans = pos, lacks_p, f"Can {X} be {a_sg}?", "no"
        work = f"Rule: all {pl} have {P}. {X} does not have {P}. So {X} is not {a_sg}."
    elif form == "u2c":
        rule, fact, q, ans = pos, lacks_p, f"Can we conclude that {X} is not {a_sg}?", "yes"
        work = f"Rule: all {pl} have {P}. {X} does not have {P}. So {X} is not {a_sg}."
    elif form == "u3":
        rule, fact, q, ans = pos, has_p, f"Can we conclude that {X} is {a_sg}?", "no"
        work = f"Rule: all {pl} have {P}. {X} has {P}, but others may have {P} too. So {X} being {a_sg} is not certain."
    elif form == "u4":
        rule, fact, q, ans = neg, is_g, f"Does {X} have {P}?", "no"
        work = f"Rule: no {pl} have {P}. {X} is {a_sg}. So {X} does not have {P}."
    elif form == "u4b":
        rule, fact, q, ans = neg, is_g, f"Can we conclude that {X} does not have {P}?", "yes"
        work = f"Rule: no {pl} have {P}. {X} is {a_sg}. So {X} does not have {P}."
    elif form == "u5":
        rule, fact, q, ans = pos, not_g, f"Can we conclude that {X} does not have {P}?", "no"
        work = f"Rule: all {pl} have {P}. {X} is not {a_sg}, so the rule says nothing about {X}."
    else:  # u6
        rule, fact, q, ans = some, is_g, f"Can we conclude that {X} has {P}?", "no"
        work = f"Rule: only some {pl} have {P}. {X} is {a_sg}, so it is not certain that {X} has {P}."

    return f"{rule} {fact} {q} {rng.choice(YN)}", work, ans, "universal_" + form


# ---------------------------------------------------------------
# Ordering from comparisons (3 or 4 people)
# ---------------------------------------------------------------
def make_ordering():
    more, less, sup_max, sup_min = rng.choice(ADJ)
    n = 3 if rng.random() < 0.6 else 4
    chain = rng.sample(NAMES, n)  # chain[0] has the most, chain[-1] the least

    facts = []
    for i in range(n - 1):
        if rng.random() < 0.5:
            facts.append(f"{chain[i]} is {more} than {chain[i + 1]}.")
        else:
            facts.append(f"{chain[i + 1]} is {less} than {chain[i]}.")
    rng.shuffle(facts)
    fact_text = " ".join(facts)
    frame = rng.choice(["{f}", "Here is what we know. {f}", "Facts: {f}", "Read these statements. {f}"])

    order = f"From {sup_max} to {sup_min}: {', '.join(chain)}."
    kind = rng.choice(["max", "min", "yn"])
    if kind == "max":
        q, ans = f"Who is the {sup_max}?", chain[0]
        work = f"{order} So the {sup_max} is {ans}."
    elif kind == "min":
        q, ans = f"Who is the {sup_min}?", chain[-1]
        work = f"{order} So the {sup_min} is {ans}."
    else:
        i, j = rng.sample(range(n), 2)
        ea, eb = chain[i], chain[j]
        q = f"Is {ea} {more} than {eb}? {rng.choice(YN)}"
        if i < j:
            ans = "yes"
            work = f"{order} {ea} comes before {eb}, so {ea} is {more} than {eb}."
        else:
            ans = "no"
            work = f"{order} {ea} comes after {eb}, so {ea} is not {more} than {eb}."
    return f"{frame.format(f=fact_text)} {q}", work, ans, "ordering_" + kind


MAKERS = [(make_conditional, 0.30), (make_universal, 0.40), (make_ordering, 0.30)]
funcs = [f for f, _ in MAKERS]
weights = [w for _, w in MAKERS]

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

counts = {"yes": 0, "no": 0}
written = skipped = rejected = 0
samples = []
with out_path.open("w", encoding="utf-8") as f:
    while written < args.n:
        question, work, answer, kind = rng.choices(funcs, weights=weights, k=1)[0]()

        # Keep yes and no exactly balanced across the whole file.
        if answer in counts:
            other = "no" if answer == "yes" else "yes"
            if counts[answer] > counts[other]:
                rejected += 1
                continue

        if norm(question) in eval_questions:
            skipped += 1
            continue

        if answer in counts:
            counts[answer] += 1
        text = f"Question: {question}\nWork: {work}\nAnswer: {answer}"
        record = {"text": text, "category": f"logic_v2_{kind}",
                  "question": question, "answer": answer}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if len(samples) < 10:
            samples.append(text)
        written += 1

print(f"Wrote {written:,} examples to {out_path}")
print(f"yes: {counts['yes']:,}  no: {counts['no']:,}  "
      f"(balance-rejected {rejected:,}, eval matches skipped {skipped})\n")
for s in samples:
    print(s)
    print("---")