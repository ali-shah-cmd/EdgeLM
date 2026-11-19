
import json
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

REPO = "open-thoughts/OpenThoughts-114k"

PREFIX = "data/"

# None = download all examples
N_DOCS = None

OUT_PATH = "data/raw/openthoughts_114k.jsonl"

BATCH_SIZE = 500


# ============================================================
# FORMAT CONVERSATION
# ============================================================

def format_conversation(system, conversations):

    parts = []

    if system:
        parts.append(f"### System\n{system}")

    for message in conversations:

        role = message.get("from", "unknown")
        content = message.get("value", "")

        if not content:
            continue

        parts.append(f"### {role}\n{content}")

    return "\n\n".join(parts)


# ============================================================
# DOWNLOAD PARQUET FILES
# ============================================================

def main():

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = sorted(
        name
        for name in HfApi().list_repo_files(
            REPO,
            repo_type="dataset"
        )
        if name.startswith(PREFIX)
        and name.endswith(".parquet")
    )

    if not files:
        raise SystemExit("No parquet files found!")

    print(f"Found {len(files)} parquet files")

    total_docs = 0

    with open(OUT_PATH, "w", encoding="utf-8") as f:

        for file_index, filename in enumerate(files, start=1):

            if N_DOCS is not None and total_docs >= N_DOCS:
                break

            print(f"\nDownloading file {file_index}/{len(files)}")
            print(filename)

            local_path = hf_hub_download(
                repo_id=REPO,
                filename=filename,
                repo_type="dataset"
            )

            pf = pq.ParquetFile(local_path)

            for batch in pf.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["system", "conversations"]
            ):

                systems = batch.column("system").to_pylist()
                conversations = batch.column("conversations").to_pylist()

                for system, conversation in zip(
                    systems,
                    conversations
                ):

                    if N_DOCS is not None and total_docs >= N_DOCS:
                        break

                    text = format_conversation(
                        system,
                        conversation
                    )

                    if not text.strip():
                        continue

                    record = {
                        "text": text
                    }

                    f.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

                    total_docs += 1

                if total_docs % 1000 < BATCH_SIZE:
                    print(f"{total_docs:,} examples saved")

                if N_DOCS is not None and total_docs >= N_DOCS:
                    break

            print(f"Finished file {file_index}")

    print("\n================================")
    print("DOWNLOAD COMPLETED")
    print("================================")

    print(f"Total examples: {total_docs:,}")
    print(f"Output file: {OUT_PATH}")

    size_mb = os.path.getsize(OUT_PATH) / (1024 ** 2)

    print(f"Output size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
