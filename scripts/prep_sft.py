import json
import os

import numpy as np
from tokenizers import Tokenizer

SRC = "data/raw/smol_smoltalk_train.jsonl"
OUT_DIR = "data/sft"
MAX_LEN = 1024

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = tok.token_to_id("<eos>")
PAD = tok.token_to_id("<pad>")
os.makedirs(OUT_DIR, exist_ok=True)


def to_pairs(messages):
    """[system?] + user/assistant alternating  ->  list of (prompt, response) texts.
    Returns None if the conversation doesn't have that shape."""
    msgs = list(messages)
    system = None
    if msgs and msgs[0]["role"] == "system":
        system = msgs[0]["content"].strip()
        msgs = msgs[1:]
    if len(msgs) < 2 or len(msgs) % 2 != 0:
        return None
    pairs = []
    for i in range(0, len(msgs), 2):
        u, a = msgs[i], msgs[i + 1]
        if u["role"] != "user" or a["role"] != "assistant":
            return None
        prompt = f"User: {u['content'].strip()}\nAssistant:"
        if i == 0 and system:
            prompt = f"System: {system}\n" + prompt
        pairs.append((prompt, " " + a["content"].strip()))
    return pairs


def build_row(pair_ids):
    """Join exchanges until the next one would not fit in MAX_LEN."""
    ids, mask, n_kept = [EOS], [0], 0
    for p_ids, r_ids in pair_ids:
        r_ids = r_ids + [EOS]
        if len(ids) + len(p_ids) + len(r_ids) > MAX_LEN:
            break
        ids += p_ids + r_ids
        mask += [0] * len(p_ids) + [1] * len(r_ids)
        n_kept += 1
    if n_kept == 0:
        return None
    return ids, mask, n_kept


n_total = sum(1 for _ in open(SRC, encoding="utf-8"))
ids_arr = np.full((n_total, MAX_LEN), PAD, dtype=np.uint16)
mask_arr = np.zeros((n_total, MAX_LEN), dtype=np.uint8)
stats = {"read": 0, "kept": 0, "bad_format": 0, "first_too_long": 0, "truncated": 0}


def process(batch_pairs):
    texts = []
    for pairs in batch_pairs:
        if pairs is not None:
            for p, r in pairs:
                texts += [p, r]
    encoded = iter(tok.encode_batch(texts))

    for pairs in batch_pairs:
        stats["read"] += 1
        if pairs is None:
            stats["bad_format"] += 1
            continue
        pair_ids = [(next(encoded).ids, next(encoded).ids) for _ in pairs]
        row = build_row(pair_ids)
        if row is None:
            stats["first_too_long"] += 1
            continue
        ids, mask, n_kept = row
        k = stats["kept"]
        ids_arr[k, :len(ids)] = ids
        mask_arr[k, :len(mask)] = mask
        stats["kept"] += 1
        if n_kept < len(pairs):
            stats["truncated"] += 1


batch = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        batch.append(to_pairs(json.loads(line)["messages"]))
        if len(batch) == 1000:
            process(batch)
            batch = []
            if stats["read"] % 50000 == 0:
                print(f"{stats['read']:,} conversations processed")
    if batch:
        process(batch)

k = stats["kept"]
np.save(f"{OUT_DIR}/ids.npy", ids_arr[:k])
np.save(f"{OUT_DIR}/mask.npy", mask_arr[:k])

real = int((ids_arr[:k] != PAD).sum())
trained = int(mask_arr[:k].sum())
print("\n" + "=" * 50)
print(f"read              : {stats['read']:,}")
print(f"kept              : {k:,}")
print(f"bad format        : {stats['bad_format']:,}")
print(f"first exchange > {MAX_LEN} tokens (dropped): {stats['first_too_long']:,}")
print(f"truncated to fewer exchanges: {stats['truncated']:,}")
print(f"real tokens       : {real:,}")
print(f"trained tokens    : {trained:,}  ({trained / real:.1%} of real tokens)")
print(f"padded tokens/epoch: {k * MAX_LEN:,}")

# ---- check one row: whole text, then only the part that is trained ----
ids0, mask0 = ids_arr[0], mask_arr[0]
n = int((ids0 != PAD).sum())
print("\nrow 0, full text (first 600 chars):")
print(tok.decode(ids0[:n].tolist(), skip_special_tokens=False)[:600])
print("\nrow 0, TRAINED part only (first 600 chars):")
print(tok.decode(ids0[:n][mask0[:n] == 1].tolist(), skip_special_tokens=False)[:600])