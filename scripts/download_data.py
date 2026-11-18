
import json
from itertools import islice

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "HuggingFaceFW/fineweb-edu"
PREFIX = "sample/10BT/"

# Set to None to extract all available documents
N_DOCS = None

OUT_PATH = "data/raw/fineweb_edu.jsonl"

# Get all parquet files
files = sorted(
    name
    for name in HfApi().list_repo_files(
        REPO, repo_type="dataset"
    )
    if name.startswith(PREFIX) and name.endswith(".parquet")
)

if not files:
    raise SystemExit("No parquet files found")

print(f"Found {len(files)} parquet files")

total_docs = 0

with open(OUT_PATH, "w", encoding="utf-8") as f:

    for file_index, filename in enumerate(files, start=1):

        if N_DOCS is not None and total_docs >= N_DOCS:
            break

        print(f"\nDownloading file {file_index}/{len(files)}")
        print(filename)

        local_path = hf_hub_download(
            REPO,
            filename,
            repo_type="dataset"
        )

        pf = pq.ParquetFile(local_path)

        for batch in pf.iter_batches(
            batch_size=1000,
            columns=["text"]
        ):
            texts = batch.column("text").to_pylist()

            for text in texts:
                if N_DOCS is not None and total_docs >= N_DOCS:
                    break

                f.write(json.dumps({"text": text}) + "\n")
                total_docs += 1

            if total_docs % 10000 < 1000:
                print(f"{total_docs:,} documents saved")

            if N_DOCS is not None and total_docs >= N_DOCS:
                break

        print(f"Finished file {file_index}")

print(f"\nTotal documents saved: {total_docs:,}")
