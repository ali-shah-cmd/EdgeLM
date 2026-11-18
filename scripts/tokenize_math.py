import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

IN_PATH = Path("data/raw/math_synth.jsonl")
OUT_PATH = Path("data/tokenized/math_synth.bin")
EOS_POSITION = "after"  # change to "after" if inspect_bin.py shows documents END with <eos>
BATCH = 20_000


def encode_and_write(tok, texts, eos, fout):
    out = []
    for enc in tok.encode_batch(texts):
        if EOS_POSITION == "before":
            out.extend([eos] + enc.ids)
        else:
            out.extend(enc.ids + [eos])
    arr = np.array(out, dtype=np.uint16)
    arr.tofile(fout)
    return len(arr)


def main():
    tok = Tokenizer.from_file("tokenizer/tokenizer.json")
    eos = tok.token_to_id("<eos>")

    total_tokens, n_docs, batch = 0, 0, []
    with open(IN_PATH, encoding="utf-8") as fin, open(OUT_PATH, "wb") as fout:
        for line in fin:
            batch.append(json.loads(line)["text"])
            if len(batch) == BATCH:
                total_tokens += encode_and_write(tok, batch, eos, fout)
                n_docs += len(batch)
                batch = []
        if batch:
            total_tokens += encode_and_write(tok, batch, eos, fout)
            n_docs += len(batch)

    print(f"{n_docs} examples -> {total_tokens} tokens ({total_tokens / n_docs:.1f} per example)")
    print(f"saved to {OUT_PATH}")

    d = np.memmap(OUT_PATH, dtype=np.uint16, mode="r")
    print("max token id:", int(d.max()), "(must be below the vocab size)")
    print()
    print(tok.decode(d[:80].tolist(), skip_special_tokens=False))


if __name__ == "__main__":
    main()