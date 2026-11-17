import argparse
import json
import os
import random
import re
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Config, SLM


# ---------------------------------------------------------------
# 1. Getting the data (parquet files straight from the Hub)
# ---------------------------------------------------------------

def pick_files(files, split, config=None):
    """Choose the parquet files that belong to one split (and one config)."""
    hits = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        parts = f.split("/")
        if config is not None and config not in parts:
            continue
        in_dir = split in parts[:-1]
        in_name = parts[-1].startswith(split + "-") or parts[-1].startswith(split + ".")
        if in_dir or in_name:
            hits.append(f)
    return sorted(hits)


def load_split(repo, split, config=None):
    api = HfApi()
    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
        except Exception as e:
            print(f"  ({repo}: could not list files at {revision or 'main'}: {type(e).__name__})")
            continue
        hits = pick_files(files, split, config)
        if hits:
            rows = []
            for name in hits:
                path = hf_hub_download(repo, name, repo_type="dataset", revision=revision)
                rows += pq.read_table(path).to_pylist()
            return rows
    raise SystemExit(f"Could not find parquet files for {repo} (split={split}, config={config})")


# ---------------------------------------------------------------
# 2. Turning each dataset into (context, choices, gold)
# ---------------------------------------------------------------

def hs_clean(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")   # WikiHow artifacts in HellaSwag
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def build_hellaswag():
    rows = load_split("Rowan/hellaswag", "validation")
    examples = []
    for r in rows:
        if str(r["label"]) == "":
            continue
        context = hs_clean(r["activity_label"] + ": " + r["ctx_a"] + " " + r["ctx_b"].capitalize())
        choices = [hs_clean(e) for e in r["endings"]]
        examples.append({"context": context, "choices": choices, "gold": int(r["label"])})
    return examples


def build_arc(config):
    rows = load_split("allenai/ai2_arc", "test", config=config)
    examples = []
    for r in rows:
        ch = r["choices"]
        if isinstance(ch, list):  # list of {"text", "label"} dicts
            texts = [c["text"] for c in ch]
            labels = [c["label"] for c in ch]
        else:                     # {"text": [...], "label": [...]}
            texts, labels = list(ch["text"]), list(ch["label"])
        if r["answerKey"] not in labels:
            continue
        examples.append({
            "context": "Question: " + r["question"] + "\nAnswer:",
            "choices": texts,
            "gold": labels.index(r["answerKey"]),
        })
    return examples


TASKS = {
    "hellaswag": build_hellaswag,
    "arc_easy": lambda: build_arc("ARC-Easy"),
    "arc_challenge": lambda: build_arc("ARC-Challenge"),
}


# ---------------------------------------------------------------
# 3. Scoring: which answer does the model find most likely?
# ---------------------------------------------------------------

def evaluate(examples, forward, enc, prefix_id, device, label):
    correct = correct_norm = 0
    t0 = time.time()
    for k, ex in enumerate(examples):
        seqs, n_cont = [], []
        for ch in ex["choices"]:
            cont_text = " " + ch
            ctx = enc(ex["context"])
            whole = enc(ex["context"] + cont_text)
            cont = whole[len(ctx):] if whole[:len(ctx)] == ctx else enc(cont_text)
            full = [prefix_id] + ctx + cont
            if len(full) > 1025:
                full = full[-1025:]
            seqs.append(full)
            n_cont.append(len(cont))

        T = max(len(s) for s in seqs) - 1
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, :len(s) - 1] = torch.tensor(s[:-1])

        with torch.no_grad():
            logp = F.log_softmax(forward(x.to(device)).float(), dim=-1)

        lls = []
        for i, s in enumerate(seqs):
            n = len(s) - 1
            targets = torch.tensor(s[1:], device=device)
            token_lp = logp[i, torch.arange(n, device=device), targets]
            lls.append(token_lp[n - min(n_cont[i], n):].sum().item())

        gold = ex["gold"]
        correct += int(max(range(len(lls)), key=lls.__getitem__) == gold)
        norm = [ll / max(1, len(ch)) for ll, ch in zip(lls, ex["choices"])]
        correct_norm += int(max(range(len(norm)), key=norm.__getitem__) == gold)

        if (k + 1) % 100 == 0:
            print(f"    {label}: {k + 1}/{len(examples)} done "
                  f"({time.time() - t0:.0f}s, acc so far {correct / (k + 1):.3f})", flush=True)

    n = len(examples)
    return correct / n, correct_norm / n


# ---------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/base_final.pt")
    parser.add_argument("--others", nargs="*", default=["HuggingFaceTB/SmolLM2-135M"])
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = args.device

    # same random questions for every model
    data = {}
    for task in args.tasks:
        print(f"loading {task} ...")
        examples = TASKS[task]()
        idx = sorted(random.Random(0).sample(range(len(examples)), min(args.n, len(examples))))
        data[task] = [examples[i] for i in idx]
        print(f"  {len(data[task])} questions (of {len(examples)})")

    models = []

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    own = SLM(Config(**ckpt["cfg"]))
    own.load_state_dict(ckpt["model"])
    own.to(device).eval()
    own_tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    models.append((f"yours (step {ckpt['step']})", lambda x: own(x),
                   lambda t: own_tok.encode(t).ids, own_tok.token_to_id("<eos>")))

    for name in args.others:
        hf_tok = AutoTokenizer.from_pretrained(name)
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
        models.append((name, (lambda m: (lambda x: m(x).logits))(hf),
                       (lambda tk: (lambda t: tk(t, add_special_tokens=False)["input_ids"]))(hf_tok),
                       hf_tok.convert_tokens_to_ids("<|endoftext|>")))

    results = {}
    for label, forward, enc, prefix_id in models:
        results[label] = {}
        for task, examples in data.items():
            print(f"\n[{label}] {task}")
            acc, acc_norm = evaluate(examples, forward, enc, prefix_id, device, label)
            results[label][task] = {"n": len(examples), "acc": acc, "acc_norm": acc_norm}
            print(f"  -> acc {acc:.3f} | acc_norm {acc_norm:.3f}")

    print("\n" + "=" * 70)
    print(f"{'model':32s} {'task':15s} {'n':>5s} {'acc':>7s} {'acc_norm':>9s}")
    for label, per_task in results.items():
        for task, r in per_task.items():
            print(f"{label:32s} {task:15s} {r['n']:5d} {r['acc']:7.3f} {r['acc_norm']:9.3f}")
    print("(random guessing is about 0.25 on all three tasks)")

    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/accuracy.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved eval_results/accuracy.json")


if __name__ == "__main__":
    main()