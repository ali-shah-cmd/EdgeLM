import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import Config, SLM

# Solved examples shown before every question. They teach the base model the
# "Question: ... Answer: ..." format. Numbers differ from the eval set.
SHOTS = [
    ("What is 3 + 4?", "7"),
    ("What is 14 * 6?", "84"),
    ("Which number is larger, 61 or 38?", "61"),
    ("What comes next: 5, 8, 11, 14, ?", "17"),
    ("Ben has 8 cups. Ben buys 5 more. How many cups does Ben have now?", "13"),
    ("All hens are birds. Sue is a hen. Is Sue a bird? Answer yes or no.", "yes"),
    ("No cows are fish. Sue is a cow. Is Sue a fish? Answer yes or no.", "no"),
]


def build_prefix():
    return "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in SHOTS)


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["cfg"])
    model = SLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt


def clean(text):
    # keep only the first line of the answer, and cut at the next "Question"
    text = text.split("\n")[0]
    text = text.split("Question")[0]
    return text.strip()


def normalize(s):
    return s.strip().lower().rstrip(".")


@torch.no_grad()
def answer(model, cfg, tok, eos_id, prefix, question, device, max_new=12):
    prompt = f"{prefix}Question: {question}\nAnswer:"
    ids = [eos_id] + tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    new_ids = []
    for _ in range(max_new):
        logits = model(x[:, -cfg.max_seq:])[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)  # greedy: always the top token
        x = torch.cat([x, next_id], dim=1)
        new_ids.append(next_id.item())
        decoded = tok.decode(new_ids)
        if "\n" in decoded or "Question" in decoded:
            break
    return clean(tok.decode(new_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", default="eval_data/math_eval.jsonl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, cfg, ckpt = load_model(args.ckpt, args.device)
    print(f"checkpoint {args.ckpt}, step {ckpt.get('step')}")

    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    eos_id = tok.token_to_id("<eos>")
    prefix = build_prefix()

    with open(args.data, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f]

    correct = defaultdict(int)
    total = defaultdict(int)
    records = []
    for p in problems:
        pred = answer(model, cfg, tok, eos_id, prefix, p["question"], args.device)
        ok = normalize(pred) == normalize(p["answer"])
        correct[p["category"]] += ok
        total[p["category"]] += 1
        records.append({**p, "prediction": pred, "correct": ok})

    print()
    for cat in total:
        print(f"{cat:14s} {correct[cat]:2d}/{total[cat]}  ({correct[cat] / total[cat]:.0%})")
    all_correct = sum(correct.values())
    print(f"{'OVERALL':14s} {all_correct}/{len(problems)}  ({all_correct / len(problems):.0%})")

    print("\nFirst 5 outputs:")
    for r in records[:5]:
        print(f"  {r['question']!r} -> {r['prediction']!r} (expected {r['answer']!r})")

    out = args.out or f"eval_results/math_{Path(args.ckpt).stem}.json"
    Path(out).parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "step": ckpt.get("step"), "records": records}, f, indent=1)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()