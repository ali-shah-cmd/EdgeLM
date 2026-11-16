import json
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer

REPO = "HuggingFaceTB/smol-smoltalk"
OUT = "data/raw/smol_smoltalk_train.jsonl"

# ---------- 1. find and download the train parquet files ----------
files = sorted(
    f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
    if f.endswith(".parquet")
)
print("parquet files in the repo:")
for f in files:
    print("  ", f)

train_files = [f for f in files if "train" in f.split("/")[-1]]
if not train_files:
    raise SystemExit("No train parquet found - paste the list above to me")

n = 0
with open(OUT, "w", encoding="utf-8") as out:
    for name in train_files:
        path = hf_hub_download(REPO, name, repo_type="dataset")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1000, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                out.write(json.dumps({"messages": messages}) + "\n")
                n += 1
print(f"\nsaved {n:,} conversations to {OUT}")

# ---------- 2. conversation shapes ----------
patterns = Counter()
sample_texts = []
first = None
with open(OUT, encoding="utf-8") as f:
    for i, line in enumerate(f):
        msgs = json.loads(line)["messages"]
        patterns[tuple(m["role"] for m in msgs)] += 1
        if first is None:
            first = msgs
        if i % 50 == 0:  # every 50th conversation as a sample
            sample_texts.append(
                "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
            )

print("\nmost common role patterns:")
for pattern, count in patterns.most_common(8):
    print(f"  {count:8,}  {pattern}")

# ---------- 3. token lengths with YOUR tokenizer ----------
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
lengths = np.array([len(e.ids) for e in tok.encode_batch(sample_texts)])
print(f"\ntoken length over {len(lengths):,} sampled conversations:")
for p in (50, 90, 95, 99):
    print(f"  p{p}: {int(np.percentile(lengths, p))}")
print(f"  fraction that fit in 1024 tokens: {(lengths <= 1024).mean():.1%}")
print(f"  average: {lengths.mean():.0f} tokens")

# ---------- 4. one example ----------
print("\nfirst conversation:")
for m in first:
    print(f"[{m['role']}] {m['content'][:300]!r}")