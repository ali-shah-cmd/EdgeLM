
# #corrected one
# import argparse
# import json
# import os
# import random
# import re
# import time

# import pyarrow.parquet as pq
# import torch
# import torch.nn.functional as F
# from huggingface_hub import HfApi, hf_hub_download
# from tokenizers import Tokenizer
# from transformers import AutoModelForCausalLM, AutoTokenizer

# from model import Config, SLM


# # ---------------------------------------------------------------
# # 1. Getting the data (parquet files straight from the Hub)
# # ---------------------------------------------------------------

# def pick_files(files, split, config=None):
#     """Choose parquet files belonging to one split/config."""
#     hits = []

#     for f in files:
#         if not f.endswith(".parquet"):
#             continue

#         parts = f.split("/")

#         if config is not None and config not in parts:
#             continue

#         in_dir = split in parts[:-1]
#         in_name = (
#             parts[-1].startswith(split + "-")
#             or parts[-1].startswith(split + ".")
#         )

#         if in_dir or in_name:
#             hits.append(f)

#     return sorted(hits)


# def load_split(repo, split, config=None):
#     """Download and load a dataset split from Hugging Face."""

#     api = HfApi()

#     for revision in (None, "refs/convert/parquet"):

#         try:
#             files = api.list_repo_files(
#                 repo,
#                 repo_type="dataset",
#                 revision=revision
#             )

#         except Exception as e:
#             print(
#                 f"  ({repo}: could not list files at "
#                 f"{revision or 'main'}: {type(e).__name__})"
#             )
#             continue

#         hits = pick_files(files, split, config)

#         if hits:
#             rows = []

#             for name in hits:
#                 print(f"  Downloading {name}...")

#                 path = hf_hub_download(
#                     repo,
#                     name,
#                     repo_type="dataset",
#                     revision=revision
#                 )

#                 rows.extend(
#                     pq.read_table(path).to_pylist()
#                 )

#             return rows

#     raise SystemExit(
#         f"Could not find parquet files for "
#         f"{repo} (split={split}, config={config})"
#     )


# # ---------------------------------------------------------------
# # 2. Turning each dataset into (context, choices, gold)
# # ---------------------------------------------------------------

# def hs_clean(text):

#     text = text.strip()
#     text = text.replace(" [title]", ". ")
#     text = re.sub(r"\[.*?\]", "", text)
#     text = re.sub(r"\s+", " ", text)

#     return text


# def build_hellaswag():

#     rows = load_split(
#         "Rowan/hellaswag",
#         "validation"
#     )

#     examples = []

#     for r in rows:

#         if r["label"] is None or str(r["label"]) == "":
#             continue

#         ctx_b = r["ctx_b"]

#         if ctx_b:
#             ctx_b = ctx_b[:1].upper() + ctx_b[1:]

#         context = hs_clean(
#             r["activity_label"]
#             + ": "
#             + r["ctx_a"]
#             + " "
#             + ctx_b
#         )

#         choices = [
#             hs_clean(e)
#             for e in r["endings"]
#         ]

#         examples.append({
#             "context": context,
#             "choices": choices,
#             "gold": int(r["label"])
#         })

#     return examples


# def build_arc(config):

#     rows = load_split(
#         "allenai/ai2_arc",
#         "test",
#         config=config
#     )

#     examples = []

#     for r in rows:

#         ch = r["choices"]

#         if isinstance(ch, list):

#             texts = [c["text"] for c in ch]
#             labels = [c["label"] for c in ch]

#         else:

#             texts = list(ch["text"])
#             labels = list(ch["label"])

#         if r["answerKey"] not in labels:
#             continue

#         examples.append({
#             "context": (
#                 "Question: "
#                 + r["question"]
#                 + "\nAnswer:"
#             ),
#             "choices": texts,
#             "gold": labels.index(r["answerKey"])
#         })

#     return examples


# TASKS = {
#     "hellaswag": build_hellaswag,
#     "arc_easy": lambda: build_arc("ARC-Easy"),
#     "arc_challenge": lambda: build_arc("ARC-Challenge"),
# }


# # ---------------------------------------------------------------
# # 3. Scoring: which answer does the model find most likely?
# # ---------------------------------------------------------------

# def evaluate(
#     examples,
#     forward,
#     enc,
#     prefix_id,
#     device,
#     label,
#     max_length=2048
# ):

#     correct = 0
#     correct_norm = 0

#     t0 = time.time()

#     for k, ex in enumerate(examples):

#         seqs = []
#         score_starts = []

#         for ch in ex["choices"]:

#             cont_text = " " + ch

#             # Tokenize context and combined text
#             ctx = enc(ex["context"])
#             whole = enc(ex["context"] + cont_text)

#             # Find the common token prefix
#             common = 0

#             for a, b in zip(ctx, whole):

#                 if a != b:
#                     break

#                 common += 1

#             # Split combined text into context and continuation
#             context_tokens = whole[:common]
#             continuation_tokens = whole[common:]

#             # Reserve space for BOS if available
#             prefix_len = 1 if prefix_id is not None else 0
#             capacity = max_length - prefix_len

#             if capacity <= 0:
#                 raise ValueError("max_length must be greater than 1")

#             # Keep the continuation whenever possible.
#             # If it alone exceeds the context window,
#             # retain its last tokens.
#             if len(continuation_tokens) > capacity:

#                 continuation_tokens = continuation_tokens[-capacity:]
#                 context_tokens = []

#             else:

#                 available_context = (
#                     capacity - len(continuation_tokens)
#                 )

#                 context_tokens = context_tokens[
#                     -available_context:
#                 ] if available_context > 0 else []

#             # Construct the actual input sequence
#             full = (
#                 ([prefix_id] if prefix_id is not None else [])
#                 + context_tokens
#                 + continuation_tokens
#             )

#             # First continuation target's logit position
#             score_start = max(
#                 0,
#                 prefix_len + len(context_tokens) - 1
#             )

#             seqs.append(full)
#             score_starts.append(score_start)

#         # Right-pad sequences
#         T = max(len(s) for s in seqs) - 1

#         if T <= 0:
#             raise ValueError("Encountered an empty evaluation sequence")

#         x = torch.zeros(
#             len(seqs),
#             T,
#             dtype=torch.long
#         )

#         for i, s in enumerate(seqs):

#             x[i, :len(s) - 1] = torch.tensor(
#                 s[:-1],
#                 dtype=torch.long
#             )

#         # Model inference
#         with torch.inference_mode():

#             logits = forward(x.to(device))

#             logp = F.log_softmax(
#                 logits.float(),
#                 dim=-1
#             )

#         lls = []
#         token_counts = []

#         for i, s in enumerate(seqs):

#             n = len(s) - 1

#             targets = torch.tensor(
#                 s[1:],
#                 device=device,
#                 dtype=torch.long
#             )

#             token_lp = logp[
#                 i,
#                 torch.arange(n, device=device),
#                 targets
#             ]

#             # Only score continuation tokens
#             start = score_starts[i]

#             continuation_lp = token_lp[start:]

#             ll = continuation_lp.sum().item()
#             count = len(continuation_lp)

#             lls.append(ll)
#             token_counts.append(count)

#         # Normal accuracy
#         gold = ex["gold"]

#         correct += int(
#             max(range(len(lls)), key=lls.__getitem__) == gold
#         )

#         # Normalized accuracy:
#         # Divide by continuation TOKEN count, not characters.
#         norm = [
#             ll / max(1, count)
#             for ll, count in zip(lls, token_counts)
#         ]

#         correct_norm += int(
#             max(range(len(norm)), key=norm.__getitem__) == gold
#         )

#         # Progress
#         if (k + 1) % 100 == 0:

#             print(
#                 f"    {label}: {k + 1}/{len(examples)} done "
#                 f"({time.time() - t0:.0f}s, "
#                 f"acc so far {correct / (k + 1):.3f})",
#                 flush=True
#             )

#     n = len(examples)

#     if n == 0:
#         return 0.0, 0.0

#     return correct / n, correct_norm / n


# # ---------------------------------------------------------------
# # 4. Model context length helper
# # ---------------------------------------------------------------

# def get_hf_max_length(model):

#     config = model.config

#     for attr in (
#         "max_position_embeddings",
#         "n_positions",
#         "max_sequence_length"
#     ):

#         value = getattr(config, attr, None)

#         if (
#             isinstance(value, int)
#             and 1 < value < 1_000_000
#         ):
#             return value

#     return 2048


# # ---------------------------------------------------------------
# # 5. Main
# # ---------------------------------------------------------------

# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--ckpt",
#         default="checkpoints/base_final.pt"
#     )

#     parser.add_argument(
#         "--others",
#         nargs="*",
#         default=["HuggingFaceTB/SmolLM2-135M"]
#     )

#     parser.add_argument(
#         "--tasks",
#         nargs="*",
#         default=list(TASKS)
#     )

#     parser.add_argument(
#         "--n",
#         type=int,
#         default=1000
#     )

#     parser.add_argument(
#         "--device",
#         default="cpu"
#     )

#     parser.add_argument(
#         "--max-length",
#         type=int,
#         default=None
#     )

#     args = parser.parse_args()

#     if args.n < 1:
#         parser.error("--n must be at least 1")

#     if args.max_length is not None and args.max_length < 2:
#         parser.error("--max-length must be at least 2")

#     unknown_tasks = set(args.tasks) - set(TASKS)

#     if unknown_tasks:
#         parser.error(
#             f"Unknown tasks: {sorted(unknown_tasks)}"
#         )

#     if args.device.startswith("cuda") and not torch.cuda.is_available():
#         parser.error("CUDA requested but not available")

#     device = torch.device(args.device)

#     print("\n========== EVALUATION CONFIGURATION ==========")
#     print(f"Device: {device}")
#     print(f"Checkpoint: {args.ckpt}")
#     print(f"Number of questions per task: {args.n}")
#     print(f"Tasks: {args.tasks}")
#     print("==============================================\n")

#     # -----------------------------------------------------------
#     # Load the same random questions for every model
#     # -----------------------------------------------------------

#     data = {}

#     for task in args.tasks:

#         print(f"Loading {task} ...")

#         examples = TASKS[task]()

#         rng = random.Random(0)

#         idx = sorted(
#             rng.sample(
#                 range(len(examples)),
#                 min(args.n, len(examples))
#             )
#         )

#         data[task] = [
#             examples[i] for i in idx
#         ]

#         print(
#             f"  {len(data[task])} questions "
#             f"(of {len(examples)})"
#         )

#     # -----------------------------------------------------------
#     # Load models
#     # -----------------------------------------------------------

#     models = []

#     print("\nLoading your SLM...")

#     ckpt = torch.load(
#         args.ckpt,
#         map_location="cpu",
#         weights_only=False
#     )

#     own = SLM(
#         Config(**ckpt["cfg"])
#     )

#     own.load_state_dict(ckpt["model"])

#     own.to(device).eval()

#     own_tok = Tokenizer.from_file(
#         "tokenizer/tokenizer.json"
#     )

#     own_bos = own_tok.token_to_id("<bos>")

#     if own_bos is None:
#         raise ValueError(
#             "Your tokenizer does not contain <bos>"
#         )

#     own_max_length = (
#         args.max_length
#         or ckpt["cfg"].get("max_seq", 2048)
#     )

#     models.append((
#         f"yours (step {ckpt.get('step', 'unknown')})",
#         lambda x: own(x),
#         lambda t: own_tok.encode(t).ids,
#         own_bos,
#         own_max_length
#     ))

#     # -----------------------------------------------------------
#     # Load Hugging Face comparison models
#     # -----------------------------------------------------------

#     for name in args.others:

#         print(f"\nLoading {name}...")

#         hf_tok = AutoTokenizer.from_pretrained(name)

#         hf = AutoModelForCausalLM.from_pretrained(
#             name,
#             torch_dtype=torch.float32
#         ).to(device).eval()

#         hf_max_length = (
#             args.max_length
#             or get_hf_max_length(hf)
#         )

#         hf_bos = hf_tok.bos_token_id

#         if hf_bos is None:
#             print(
#                 f"WARNING: {name} has no BOS token. "
#                 "Scoring will start from its available context."
#             )

#         models.append((
#             name,
#             lambda x, m=hf: m(x).logits,
#             lambda t, tk=hf_tok: tk(
#                 t,
#                 add_special_tokens=False
#             )["input_ids"],
#             hf_bos,
#             hf_max_length
#         ))

#     # -----------------------------------------------------------
#     # Run evaluation
#     # -----------------------------------------------------------

#     results = {}

#     for label, forward, enc, prefix_id, max_length in models:

#         results[label] = {}

#         print(f"\n========== MODEL: {label} ==========")
#         print(f"Maximum sequence length: {max_length}")

#         for task, examples in data.items():

#             print(f"\n[{label}] {task}")

#             acc, acc_norm = evaluate(
#                 examples,
#                 forward,
#                 enc,
#                 prefix_id,
#                 device,
#                 label,
#                 max_length=max_length
#             )

#             results[label][task] = {
#                 "n": len(examples),
#                 "acc": acc,
#                 "acc_norm": acc_norm
#             }

#             print(
#                 f"  -> acc: {acc:.3f} "
#                 f"| acc_norm: {acc_norm:.3f}"
#             )

#     # -----------------------------------------------------------
#     # Results summary
#     # -----------------------------------------------------------

#     print("\n" + "=" * 75)

#     print(
#         f"{'Model':32s} "
#         f"{'Task':18s} "
#         f"{'N':>6s} "
#         f"{'Accuracy':>10s} "
#         f"{'Acc_norm':>10s}"
#     )

#     print("-" * 75)

#     for label, per_task in results.items():

#         for task, r in per_task.items():

#             print(
#                 f"{label:32s} "
#                 f"{task:18s} "
#                 f"{r['n']:6d} "
#                 f"{r['acc']:10.3f} "
#                 f"{r['acc_norm']:10.3f}"
#             )

#     print("-" * 75)

#     # Random guessing baseline
#     print("\nRandom guessing baselines:")

#     for task, examples in data.items():

#         baseline = sum(
#             1 / len(ex["choices"])
#             for ex in examples
#         ) / len(examples)

#         print(f"  {task}: {baseline:.3f}")

#     # -----------------------------------------------------------
#     # Save results
#     # -----------------------------------------------------------

#     os.makedirs(
#         "eval_results",
#         exist_ok=True
#     )

#     output_path = "eval_results/accuracy.json"

#     with open(
#         output_path,
#         "w",
#         encoding="utf-8"
#     ) as f:

#         json.dump(
#             results,
#             f,
#             indent=2
#         )

#     print(f"\nSaved results to {output_path}")


# if __name__ == "__main__":
#     main()




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


# ===============================================================
# 1. LOAD DATASETS
# ===============================================================

def pick_files(files, split, config=None):
    hits = []

    for f in files:
        if not f.endswith(".parquet"):
            continue

        parts = f.split("/")

        if config is not None and config not in parts:
            continue

        in_dir = split in parts[:-1]
        in_name = (
            parts[-1].startswith(split + "-")
            or parts[-1].startswith(split + ".")
        )

        if in_dir or in_name:
            hits.append(f)

    return sorted(hits)


def load_split(repo, split, config=None):
    api = HfApi()

    for revision in (None, "refs/convert/parquet"):
        try:
            files = api.list_repo_files(
                repo,
                repo_type="dataset",
                revision=revision
            )
        except Exception as e:
            print(
                f"({repo}: could not list files at "
                f"{revision or 'main'}: {type(e).__name__})"
            )
            continue

        hits = pick_files(files, split, config)

        if hits:
            rows = []

            for name in hits:
                print(f"Downloading {name}...")

                path = hf_hub_download(
                    repo,
                    name,
                    repo_type="dataset",
                    revision=revision
                )

                rows.extend(pq.read_table(path).to_pylist())

            return rows

    raise SystemExit(
        f"Could not find parquet files for "
        f"{repo} (split={split}, config={config})"
    )


# ===============================================================
# 2. PREPARE BENCHMARK QUESTIONS
# ===============================================================

def hs_clean(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def build_hellaswag():
    rows = load_split(
        "Rowan/hellaswag",
        "validation"
    )

    examples = []

    for r in rows:
        if r["label"] is None or str(r["label"]) == "":
            continue

        ctx_b = r["ctx_b"]

        if ctx_b:
            ctx_b = ctx_b[:1].upper() + ctx_b[1:]

        context = hs_clean(
            r["activity_label"]
            + ": "
            + r["ctx_a"]
            + " "
            + ctx_b
        )

        choices = [
            hs_clean(e)
            for e in r["endings"]
        ]

        examples.append({
            "context": context,
            "choices": choices,
            "gold": int(r["label"])
        })

    return examples


def build_arc(config):
    rows = load_split(
        "allenai/ai2_arc",
        "test",
        config=config
    )

    examples = []

    for r in rows:
        ch = r["choices"]

        if isinstance(ch, list):
            texts = [c["text"] for c in ch]
            labels = [c["label"] for c in ch]
        else:
            texts = list(ch["text"])
            labels = list(ch["label"])

        if r["answerKey"] not in labels:
            continue

        examples.append({
            "context": (
                "Question: "
                + r["question"]
                + "\nAnswer:"
            ),
            "choices": texts,
            "gold": labels.index(r["answerKey"])
        })

    return examples


TASKS = {
    "hellaswag": build_hellaswag,
    "arc_easy": lambda: build_arc("ARC-Easy"),
    "arc_challenge": lambda: build_arc("ARC-Challenge"),
}


# ===============================================================
# 3. EVALUATION
# ===============================================================

def evaluate(
    examples,
    forward,
    enc,
    prefix_id,
    device,
    label,
    max_length=2048
):
    correct = 0
    correct_norm = 0

    t0 = time.time()

    for k, ex in enumerate(examples):
        seqs = []
        score_starts = []

        for ch in ex["choices"]:
            cont_text = " " + ch

            # Tokenize context and combined text
            ctx = enc(ex["context"])
            whole = enc(ex["context"] + cont_text)

            # Find common token prefix
            common = 0

            for a, b in zip(ctx, whole):
                if a != b:
                    break
                common += 1

            context_tokens = whole[:common]
            continuation_tokens = whole[common:]

            # Reserve space for BOS
            prefix_len = 1 if prefix_id is not None else 0
            capacity = max_length - prefix_len

            if capacity <= 0:
                raise ValueError(
                    "max_length must be greater than 1"
                )

            # Keep continuation tokens whenever possible
            if len(continuation_tokens) > capacity:
                continuation_tokens = continuation_tokens[-capacity:]
                context_tokens = []
            else:
                available_context = (
                    capacity - len(continuation_tokens)
                )

                context_tokens = (
                    context_tokens[-available_context:]
                    if available_context > 0
                    else []
                )

            # Construct input sequence
            full = (
                ([prefix_id] if prefix_id is not None else [])
                + context_tokens
                + continuation_tokens
            )

            # Position predicting the first continuation token
            score_start = max(
                0,
                prefix_len + len(context_tokens) - 1
            )

            seqs.append(full)
            score_starts.append(score_start)

        # Right padding
        T = max(len(s) for s in seqs) - 1

        if T <= 0:
            raise ValueError(
                "Encountered an empty evaluation sequence"
            )

        x = torch.zeros(
            len(seqs),
            T,
            dtype=torch.long
        )

        for i, s in enumerate(seqs):
            x[i, :len(s) - 1] = torch.tensor(
                s[:-1],
                dtype=torch.long
            )

        # Model inference
        with torch.inference_mode():
            logits = forward(x.to(device))

            logp = F.log_softmax(
                logits.float(),
                dim=-1
            )

        lls = []
        token_counts = []

        for i, s in enumerate(seqs):
            n = len(s) - 1

            targets = torch.tensor(
                s[1:],
                device=device,
                dtype=torch.long
            )

            token_lp = logp[
                i,
                torch.arange(n, device=device),
                targets
            ]

            # Score continuation tokens only
            start = score_starts[i]
            continuation_lp = token_lp[start:]

            ll = continuation_lp.sum().item()
            count = len(continuation_lp)

            lls.append(ll)
            token_counts.append(count)

        # Normal accuracy
        gold = ex["gold"]

        correct += int(
            max(range(len(lls)), key=lls.__getitem__) == gold
        )

        # Length-normalized accuracy
        norm = [
            ll / max(1, count)
            for ll, count in zip(lls, token_counts)
        ]

        correct_norm += int(
            max(range(len(norm)), key=norm.__getitem__) == gold
        )

        # Progress
        if (k + 1) % 100 == 0:
            print(
                f"    {label}: {k + 1}/{len(examples)} done "
                f"({time.time() - t0:.0f}s, "
                f"acc so far {correct / (k + 1):.3f})",
                flush=True
            )

    n = len(examples)

    if n == 0:
        return 0.0, 0.0

    return correct / n, correct_norm / n


# ===============================================================
# 4. MODEL CONTEXT LENGTH
# ===============================================================

def get_hf_max_length(model):
    config = model.config

    for attr in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length"
    ):
        value = getattr(config, attr, None)

        if (
            isinstance(value, int)
            and 1 < value < 1_000_000
        ):
            return value

    return 2048


# ===============================================================
# 5. MAIN
# ===============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        default="checkpoints/base_sft_best.pt"
    )

    parser.add_argument(
        "--others",
        nargs="*",
        default=["HuggingFaceTB/SmolLM2-135M"]
    )

    parser.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASKS)
    )

    # None means evaluate ALL available questions
    parser.add_argument(
        "--n",
        type=int,
        default=None
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        default="<eos>"
    )


    args = parser.parse_args()

    if args.n is not None and args.n < 1:
        parser.error("--n must be at least 1")

    if args.max_length is not None and args.max_length < 2:
        parser.error("--max-length must be at least 2")

    unknown_tasks = set(args.tasks) - set(TASKS)

    if unknown_tasks:
        parser.error(
            f"Unknown tasks: {sorted(unknown_tasks)}"
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    device = torch.device(args.device)

    print("\n========== EVALUATION CONFIGURATION ==========")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    print(
        f"Number of questions per task: "
        f"{args.n if args.n is not None else 'ALL'}"
    )

    print(f"Tasks: {args.tasks}")
    print("==============================================\n")

    # -----------------------------------------------------------
    # Load the same questions for every model
    # -----------------------------------------------------------

    data = {}

    for task in args.tasks:
        print(f"Loading {task}...")

        examples = TASKS[task]()

        rng = random.Random(0)

        if args.n is not None:
            idx = sorted(
                rng.sample(
                    range(len(examples)),
                    min(args.n, len(examples))
                )
            )
        else:
            # Evaluate every available example
            idx = list(range(len(examples)))

        data[task] = [
            examples[i] for i in idx
        ]

        print(
            f"  {len(data[task])} questions "
            f"(of {len(examples)})"
        )

    # -----------------------------------------------------------
    # Load your SLM
    # -----------------------------------------------------------

    models = []

    print("\nLoading your SLM...")

    ckpt = torch.load(
        args.ckpt,
        map_location="cpu",
        weights_only=False
    )

    own = SLM(
        Config(**ckpt["cfg"])
    )

    own.load_state_dict(ckpt["model"])
    own.to(device).eval()

    own_tok = Tokenizer.from_file(
        "tokenizer/tokenizer.json"
    )

    own_bos = own_tok.token_to_id(args.prefix)

    if own_bos is None:
        raise ValueError(
            f"Your tokenizer does not contain {args.prefix}"
        )

    own_max_length = (
        args.max_length
        or ckpt["cfg"].get("max_seq", 2048)
    )

    models.append((
        f"yours (step {ckpt.get('step', 'unknown')})",
        lambda x: own(x),
        lambda t: own_tok.encode(t).ids,
        own_bos,
        own_max_length
    ))

    # -----------------------------------------------------------
    # Load comparison models
    # -----------------------------------------------------------

    for name in args.others:
        print(f"\nLoading {name}...")

        hf_tok = AutoTokenizer.from_pretrained(name)

        hf = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.float32
        ).to(device).eval()

        hf_max_length = (
            args.max_length
            or get_hf_max_length(hf)
        )

        hf_bos = hf_tok.bos_token_id

        if hf_bos is None:
            print(
                f"WARNING: {name} has no BOS token. "
                "Scoring will start from its available context."
            )

        models.append((
            name,
            lambda x, m=hf: m(x).logits,
            lambda t, tk=hf_tok: tk(
                t,
                add_special_tokens=False
            )["input_ids"],
            hf_bos,
            hf_max_length
        ))

    # -----------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------

    results = {}

    for label, forward, enc, prefix_id, max_length in models:
        results[label] = {}

        print(f"\n========== MODEL: {label} ==========")
        print(f"Maximum sequence length: {max_length}")

        for task, examples in data.items():
            print(f"\n[{label}] {task}")

            acc, acc_norm = evaluate(
                examples,
                forward,
                enc,
                prefix_id,
                device,
                label,
                max_length=max_length
            )

            results[label][task] = {
                "n": len(examples),
                "acc": acc,
                "acc_norm": acc_norm
            }

            print(
                f"  -> acc: {acc:.3f} "
                f"| acc_norm: {acc_norm:.3f}"
            )

    # -----------------------------------------------------------
    # Results summary
    # -----------------------------------------------------------

    print("\n" + "=" * 75)

    print(
        f"{'Model':32s} "
        f"{'Task':18s} "
        f"{'N':>6s} "
        f"{'Accuracy':>10s} "
        f"{'Acc_norm':>10s}"
    )

    print("-" * 75)

    for label, per_task in results.items():
        for task, r in per_task.items():
            print(
                f"{label:32s} "
                f"{task:18s} "
                f"{r['n']:6d} "
                f"{r['acc']:10.3f} "
                f"{r['acc_norm']:10.3f}"
            )

    print("-" * 75)

    # Random guessing baseline
    print("\nRandom guessing baselines:")

    for task, examples in data.items():
        if not examples:
            print(f"  {task}: N/A (no examples)")
            continue

        baseline = sum(
            1 / len(ex["choices"])
            for ex in examples
        ) / len(examples)

        print(f"  {task}: {baseline:.3f}")

    # -----------------------------------------------------------
    # Save results
    # -----------------------------------------------------------

    os.makedirs(
        "eval_results",
        exist_ok=True
    )

    output_path = args.out

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()