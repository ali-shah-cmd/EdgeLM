import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get project root directory
ROOT = Path(__file__).resolve().parents[1]

# Add project root to Python module search path
sys.path.insert(0, str(ROOT))

from src.eval_reasoning import extract_answer, normalize_expected


# Hand-written worked examples (NOT taken from the eval set).
# Prompts for arithmetic questions end with "Answer:", so they get the direct examples.
DIRECT_SHOTS = (
    "Question: Calculate the total when 47 is increased by 36. Give only the integer.\nAnswer: 83\n\n"
    "Question: Which is the greater value, 215 or 152? Respond with the greater number only.\nAnswer: 215\n\n"
    "Question: Continue the pattern by one value: 5, 9, 13, 17. What value comes next? Give only the number.\nAnswer: 21\n\n"
)

# Prompts for word-problem and logic questions end with "Work:", so they get these.
WORK_SHOTS = (
    "Question: Tom had 120 stamps and bought 45 more. How many stamps does Tom have now?\nWork: 120 + 45 = 165\nAnswer: 165\n\n"
    "Question: Every pilot in the club must have a licence. Ana is a pilot in the club. Must Ana have a licence? Answer yes or no.\nWork: The rule applies to every member.\nAnswer: yes\n\n"
    "Question: If the light is on, the room is warm. The room is warm. Can we conclude that the light is on? Answer yes or no.\nWork: The rule does not work in reverse.\nAnswer: no\n\n"
    "Question: Consider these comparisons: Kim is taller than Lou, and Lou is taller than Max. Who is the shortest?\nWork: Kim > Lou > Max in height\nAnswer: Max\n\n"
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
parser.add_argument("--data", default="eval_data/reasoning_eval.jsonl")
parser.add_argument("--max-new-tokens", type=int, default=64)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output", required=True)
parser.add_argument("--shots", action="store_true",
                    help="Prepend worked examples to every prompt (few-shot).")
args = parser.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model).eval()

with open(ROOT / args.data, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if args.limit:
    counts, picked = defaultdict(int), []
    for r in rows:
        if counts[r["category"]] < args.limit:
            picked.append(r)
            counts[r["category"]] += 1
    rows = picked

totals = defaultdict(lambda: [0, 0])   # category -> [correct, total]
results = []

for i, row in enumerate(rows, 1):
    prompt = row["prompt"]
    if args.shots:
        # Prompts ending in "Work:" are word/logic questions; the rest are direct arithmetic.
        prompt = (WORK_SHOTS if prompt.endswith("Work:") else DIRECT_SHOTS) + prompt

    ids = [tok.eos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids])
    with torch.inference_mode():
        out = model.generate(
            x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    generated = tok.decode(out[0, len(ids):], skip_special_tokens=True)
    predicted = extract_answer(generated, row["answer"])
    correct = normalize_expected(predicted) == normalize_expected(row["answer"])
    totals[row["category"]][0] += int(correct)
    totals[row["category"]][1] += 1
    results.append({
        **row,
        "model": args.model,
        "shots": args.shots,
        "generated": generated,
        "predicted": predicted,
        "correct": correct,
    })
    if i % 20 == 0:
        print(f"{i}/{len(rows)}", flush=True)

print(f"\nModel: {args.model} | few-shot: {args.shots}")
for cat, (c, n) in sorted(totals.items()):
    print(f"{cat:16s}: {c}/{n} = {100 * c / n:.1f}%")
all_c = sum(c for c, _ in totals.values())
all_n = sum(n for _, n in totals.values())
print(f"OVERALL: {all_c}/{all_n} = {100 * all_c / all_n:.1f}%")

out_path = ROOT / args.output
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Final line: run summary (marked with "type": "summary" so loaders can skip it).
    summary = {
        "type": "summary",
        "settings": vars(args),
        "overall": {
            "correct": all_c,
            "total": all_n,
            "accuracy": 100 * all_c / all_n,
        },
        "categories": {
            cat: {"correct": c, "total": n, "accuracy": 100 * c / n}
            for cat, (c, n) in totals.items()
        },
    }
    f.write(json.dumps(summary, ensure_ascii=False) + "\n")

print(f"Saved to: {out_path}")