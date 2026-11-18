import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# Tokenizes JSONL files with the EXISTING tokenizer.
# It does not train or save a tokenizer.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL path")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument("--batch", type=int, default=20_000)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    eos = tok.token_to_id("<eos>")
    if eos is None:
        raise ValueError("Tokenizer does not contain <eos>.")

    total_tokens = 0
    n_docs = 0
    batch = []

    def flush(texts, fout):
        nonlocal total_tokens, n_docs
        encoded = tok.encode_batch(texts)
        values = []
        for enc in encoded:
            # Existing FineWeb train.bin puts <eos> AFTER each document.
            values.extend(enc.ids)
            values.append(eos)
        arr = np.asarray(values, dtype=np.uint16)
        if arr.size and int(arr.max()) >= tok.get_vocab_size():
            raise ValueError("Token ID is outside the tokenizer vocabulary.")
        arr.tofile(fout)
        total_tokens += len(arr)
        n_docs += len(texts)

    with in_path.open(encoding="utf-8") as fin, out_path.open("wb") as fout:
        for line_no, line in enumerate(fin, 1):
            row = json.loads(line)
            if "text" not in row or not isinstance(row["text"], str):
                raise ValueError(f"Line {line_no} has no string 'text' field.")
            batch.append(row["text"])
            if len(batch) >= args.batch:
                flush(batch, fout)
                batch = []
        if batch:
            flush(batch, fout)

    print(f"{n_docs:,} examples -> {total_tokens:,} tokens "
          f"({total_tokens / max(n_docs, 1):.1f} per example)")
    print(f"saved to {out_path}")
    data = np.memmap(out_path, dtype=np.uint16, mode="r")
    print(f"max token id: {int(data.max())}; vocab size: {tok.get_vocab_size()}")
    print("decoded start:")
    print(tok.decode(data[:100].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()
