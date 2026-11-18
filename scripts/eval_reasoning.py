import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import Config, SLM


def extract_answer(generated_text, expected):
    # Prefer the model's first explicit Answer: line. The first match is the
    # answer to OUR question; later ones can belong to extra examples the
    # model invents if it fails to stop.
    generated_text = re.split(r"\n\s*Question\s*:", generated_text)[0]
    matches = re.findall(
        r"(?im)^\s*Answer\s*:\s*(.*)$",
        generated_text
    )
    if matches:
        raw = matches[0].strip()
    else:
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        raw = lines[0] if lines else generated_text.strip()

    expected = str(expected).strip()

    if expected.lower() in ("yes", "no"):
        found = re.findall(r"\b(yes|no)\b", raw.lower())
        return found[-1] if found else raw.lower().strip(" \t\r\n.,!?")

    if re.fullmatch(r"-?\d+", expected):
        numbers = re.findall(r"-?\d[\d,]*", raw)
        if numbers:
            return numbers[-1].replace(",", "")
        return raw.strip()

    # Name-style answers: remove common answer prefixes, then take first word.
    cleaned = re.sub(
        r"(?i)^(the answer is|the tallest is|the slowest is|it is|answer is)\s+",
        "",
        raw
    ).strip()
    words = re.findall(r"[A-Za-z]+", cleaned)
    return words[0].lower() if words else cleaned.lower()


def normalize_expected(value):
    return re.sub(r"[^a-z0-9-]", "", str(value).lower())


@torch.inference_mode()
def generate(model, tokenizer, prompt, device, max_seq, max_new_tokens, prefix_id, eos_id):
    ids = tokenizer.encode(prompt).ids
    if prefix_id is not None:
        ids = [prefix_id] + ids

    if not ids:
        raise ValueError("Tokenizer produced an empty prompt.")

    if len(ids) >= max_seq:
        ids = ids[-(max_seq - 1):]

    input_ids = list(ids)
    generated_ids = []
    allowed_new = min(max_new_tokens, max_seq - len(input_ids))

    for _ in range(allowed_new):
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        logits = model(x)

        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "SLM.forward() did not return a Tensor of logits. "
                "Please share model.py so the evaluator can match its output API."
            )

        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if eos_id is not None and next_id == eos_id:
            break

        generated_ids.append(next_id)
        input_ids.append(next_id)

        if len(input_ids) >= max_seq:
            break

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate EdgeLM on fixed arithmetic, word-problem, and logic questions."
    )
    parser.add_argument("--ckpt", default="checkpoints/base_final.pt")
    parser.add_argument("--data", default="eval_data/reasoning_eval.jsonl")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional maximum number of examples per category. Default: all.")
    parser.add_argument("--show", type=int, default=5,
                        help="Number of incorrect examples to print per category.")
    parser.add_argument("--output", default=None,
                        help="Optional output JSONL path. Default is generated from checkpoint name and step.")
    parser.add_argument("--prefix", choices=["none", "bos", "eos"], default="eos",
                        help="Token placed before the prompt. Training data has <eos> between examples.")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive.")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but CUDA is not available.")

    ckpt_path = ROOT / args.ckpt
    data_path = ROOT / args.data
    device = torch.device(args.device)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Evaluation data not found: {data_path}\n"
            "Run: python scripts\\make_reasoning_eval.py"
        )

    rows = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if args.limit is not None:
        selected = []
        counts = defaultdict(int)
        for row in rows:
            cat = row["category"]
            if counts[cat] < args.limit:
                selected.append(row)
                counts[cat] += 1
        rows = selected

    if not rows:
        raise ValueError("Evaluation dataset contains no examples.")

    print("\n========== EDGE LM REASONING EVALUATION ==========")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Evaluation file: {data_path}")
    print(f"Device: {device}")
    print(f"Questions: {len(rows)}")
    print("==================================================")

    print("\nLoading checkpoint...")
    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False
    )

    if "model" not in checkpoint or "cfg" not in checkpoint:
        raise KeyError("Checkpoint must contain 'model' and 'cfg'.")

    config = Config(**checkpoint["cfg"])
    model = SLM(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tokenizer_path = ROOT / "tokenizer" / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    prefix_id = {"none": None, "bos": bos_id, "eos": eos_id}[args.prefix]

    max_seq = int(checkpoint["cfg"].get("max_seq", 2048))
    step = checkpoint.get("step", "unknown")

    print(f"Checkpoint step: {step}")
    print(f"Model context length: {max_seq}")
    print(f"Prompt prefix token: {args.prefix} (id {prefix_id})")
    print("Model loaded. Starting evaluation...\n")

    results = []
    totals = defaultdict(lambda: {"correct": 0, "total": 0})
    sub_totals = defaultdict(lambda: {"correct": 0, "total": 0})
    wrong_examples = defaultdict(list)
    yes_stats = defaultdict(lambda: {"pred_yes": 0, "true_yes": 0, "n": 0})

    start = time.time()

    for index, row in enumerate(rows, start=1):
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=row["prompt"],
            device=device,
            max_seq=max_seq,
            max_new_tokens=args.max_new_tokens,
            prefix_id=prefix_id,
            eos_id=eos_id
        )

        predicted = extract_answer(generated, row["answer"])
        expected = normalize_expected(row["answer"])
        actual = normalize_expected(predicted)
        correct = actual == expected

        category = row["category"]
        subcategory = row.get("subcategory", "unknown")

        # Track how often the model says "yes" on yes/no questions.
        if str(row["answer"]).lower() in ("yes", "no"):
            s = yes_stats[(category, subcategory)]
            s["n"] += 1
            s["pred_yes"] += int(actual == "yes")
            s["true_yes"] += int(expected == "yes")

        totals[category]["total"] += 1
        totals[category]["correct"] += int(correct)
        sub_totals[(category, subcategory)]["total"] += 1
        sub_totals[(category, subcategory)]["correct"] += int(correct)

        result = {
            "category": category,
            "subcategory": subcategory,
            "question": row["question"],
            "expected": row["answer"],
            "generated": generated,
            "predicted": predicted,
            "correct": correct,
        }
        results.append(result)

        if not correct and len(wrong_examples[category]) < args.show:
            wrong_examples[category].append(result)

        if index % 10 == 0 or index == len(rows):
            elapsed = time.time() - start
            print(f"Progress: {index}/{len(rows)} | elapsed {elapsed:.1f}s", flush=True)

    print("\n================ CATEGORY RESULTS ================")
    overall_correct = sum(v["correct"] for v in totals.values())
    overall_total = sum(v["total"] for v in totals.values())

    for category in sorted(totals):
        item = totals[category]
        accuracy = 100.0 * item["correct"] / max(1, item["total"])
        print(f"{category:16s}: {item['correct']:3d}/{item['total']:3d} = {accuracy:6.2f}%")

    print("\nSubcategory results:")
    for (category, subcategory), item in sorted(sub_totals.items()):
        accuracy = 100.0 * item["correct"] / max(1, item["total"])
        print(f"  {category:14s} / {subcategory:14s}: "
              f"{item['correct']:3d}/{item['total']:3d} = {accuracy:6.2f}%")

    print("\nYes/no balance (predicted yes vs true yes):")
    for (cat, sub), s in sorted(yes_stats.items()):
        print(f"  {cat:14s} / {sub:14s}: "
              f"predicted yes {s['pred_yes']}/{s['n']}, true yes {s['true_yes']}/{s['n']}")

    overall_accuracy = 100.0 * overall_correct / max(1, overall_total)
    print(f"\nOVERALL: {overall_correct}/{overall_total} = {overall_accuracy:.2f}%")
    print(f"Total evaluation time: {time.time() - start:.1f}s")

    for category in sorted(wrong_examples):
        print(f"\n--- Incorrect examples: {category} ---")
        for item in wrong_examples[category]:
            print(f"Q: {item['question']}")
            print(f"Expected: {item['expected']}")
            print(f"Predicted: {item['predicted']}")
            print(f"Generated: {item['generated'][:300]!r}\n")

    if args.output:
        output_path = ROOT / args.output
    else:
        output_path = ROOT / "eval_results" / f"reasoning_{ckpt_path.stem}_step_{step}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"\nDetailed predictions saved to: {output_path}")


if __name__ == "__main__":
    main()