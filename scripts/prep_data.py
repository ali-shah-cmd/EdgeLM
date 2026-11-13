import json
from itertools import islice

import numpy as np
from tokenizers import Tokenizer

SRC = "data/raw/fineweb_edu.jsonl"
TRAIN_OUT = "data/tokenized/train.bin"
VAL_OUT = "data/tokenized/val.bin"

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = tok.token_to_id("<eos>")


def read_docs(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


docs = read_docs(SRC)
doc_id = 0
n_train = 0
n_val = 0

with open(TRAIN_OUT, "wb") as f_train, open(VAL_OUT, "wb") as f_val:
    while True:
        batch = list(islice(docs, 1000))
        if not batch:
            break
        for enc in tok.encode_batch(batch):
            arr = np.array(enc.ids + [EOS], dtype=np.uint16)
            if doc_id % 100 == 0:
                arr.tofile(f_val)
                n_val += len(arr)
            else:
                arr.tofile(f_train)
                n_train += len(arr)
            doc_id += 1
        print(f"{doc_id:,} documents processed")

print(f"train tokens: {n_train:,}")
print(f"val tokens  : {n_val:,}")

data = np.memmap(TRAIN_OUT, dtype=np.uint16, mode="r")
print("max token id:", int(data.max()))
print("--- first 120 tokens decoded ---")
print(tok.decode(data[:120].tolist(), skip_special_tokens=False))