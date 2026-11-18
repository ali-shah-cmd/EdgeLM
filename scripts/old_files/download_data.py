import json
from itertools import islice

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT/"
N_DOCS = 100_000
OUT_PATH = "data/raw/fineweb_edu.jsonl"

files = sorted(
    name
    for name in HfApi().list_repo_files(REPO, repo_type="dataset")
    if name.startswith(PREFIX) and name.endswith(".parquet")
)
if not files:
    raise SystemExit(f"No parquet files found under {PREFIX}")
print(f"found {len(files)} files, using {files[0]}")

local_path = hf_hub_download(REPO, files[0], repo_type="dataset")
print("downloaded to:", local_path)

pf = pq.ParquetFile(local_path)


def all_texts():
    for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
        yield from batch.column("text").to_pylist()


n = 0
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for n, text in enumerate(islice(all_texts(), N_DOCS), start=1):
        f.write(json.dumps({"text": text}) + "\n")
        if n % 10000 == 0:
            print(f"{n:,} documents saved")

print(f"saved {n:,} documents")
if n < N_DOCS:
    print("WARNING: fewer documents than requested")