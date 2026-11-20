import argparse, os, json, time
from huggingface_hub import list_repo_files
from datasets import load_dataset
from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders

REPO = "EleutherAI/SmolLM2-135M-100B"
MAX_CHARS = 100_000          # cap per document; the data prep must use the same cap
SPECIALS = ["<pad>", "<bos>", "<eos>", "<think>", "</think>"]   # ids 0-4, same order as the old tokenizer

p = argparse.ArgumentParser()
p.add_argument("--out_dir", default="tokenizer32k")
p.add_argument("--vocab_size", type=int, default=32768)
p.add_argument("--max_mb", type=int, default=600, help="stop after about this many MB of text")
p.add_argument("--group_digits", action="store_true", help="keep numbers as chunks instead of single digits")
args = p.parse_args()

# 1. Which files to use
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
tok_files = files[-3:-1]      # the two files before the last one (846, 847)
val_file = files[-1]          # 848, never used for training
print("tokenizer files:", tok_files, "| validation file:", val_file)

# 2. Stream the text, one parquet file at a time (nothing is saved to disk)
def stream_texts(fname):
    ds = load_dataset("parquet", data_files=f"hf://datasets/{REPO}/{fname}",
                      split="train", streaming=True)
    for row in ds:
        yield row["text"][:MAX_CHARS]

stats = {"chars": 0, "docs": 0}
max_chars = args.max_mb * 1_000_000

def corpus():
    for fname in tok_files:
        for text in stream_texts(fname):
            yield text
            stats["chars"] += len(text)
            stats["docs"] += 1
            if stats["chars"] >= max_chars:
                return

# 3. Build the tokenizer: split rule + byte-level BPE
digit = r"\p{N}+" if args.group_digits else r"\p{N}"      # \p{N} alone = one digit at a time
pattern = (r"'s|'t|'re|'ve|'m|'ll|'d"
           r"| ?\p{L}+"
           r"| ?" + digit +
           r"| ?[^\s\p{L}\p{N}]+"
           r"|\s+(?!\S)|\s+")

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(pattern), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=SPECIALS,                              # listed first, so they get ids 0-4
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), # all 256 bytes, so nothing is ever unknown
    min_frequency=2,
    show_progress=True,
)

# 4. Train
print(f"Starting tokenizer training on about {args.max_mb} MB of text...")
t0 = time.time()
tokenizer.train_from_iterator(corpus(), trainer=trainer)
elapsed = time.time() - t0
print(f"Trained on {stats['docs']:,} docs, {stats['chars'] / 1e6:.0f}M characters, in {elapsed / 60:.1f} minutes")

# 5. Checks
for i, t in enumerate(SPECIALS):
    assert tokenizer.token_to_id(t) == i, f"{t} is not id {i}"
assert tokenizer.get_vocab_size() == args.vocab_size
print("Special token ids:", {t: tokenizer.token_to_id(t) for t in SPECIALS})

for text in ["347 + 285 = 632",
             "In 2024 the price rose 12.5% to $1,299.",
             "def add(a, b):\n    return a + b\n",
             "The cat sat on the mat. <think>2 + 2 = 4</think>"]:
    enc = tokenizer.encode(text)
    print(enc.tokens)
    assert tokenizer.decode(enc.ids, skip_special_tokens=False) == text   # must round-trip exactly

assert tokenizer.token_to_id("<think>") in tokenizer.encode("a <think> b").ids   # specials stay single tokens

# 6. Size stats on the held-out file
texts = []
for t in stream_texts(val_file):
    texts.append(t)
    if len(texts) >= 2000:
        break
encs = tokenizer.encode_batch(texts)
n_chars = sum(len(t) for t in texts)
n_tok = sum(len(e.ids) for e in encs)
chars_per_token = n_chars / n_tok
print(f"chars per token: {chars_per_token:.2f} | tokens per doc: {n_tok / len(texts):.0f}")
DOCS_PER_FILE = 118_000      # about 100.58M docs / 849 files
print(f"estimated tokens per parquet file: {n_tok / len(texts) * DOCS_PER_FILE / 1e6:.0f}M")

# 7. Save the tokenizer and a record of how it was built
os.makedirs(args.out_dir, exist_ok=True)
tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

metadata = {
    "vocab_size": tokenizer.get_vocab_size(),
    "special_tokens": {t: tokenizer.token_to_id(t) for t in SPECIALS},
    "algorithm": "ByteLevel BPE",
    "digit_splitting": not args.group_digits,
    "max_doc_chars": MAX_CHARS,
    "train_files": tok_files,
    "train_docs": stats["docs"],
    "train_chars": stats["chars"],
    "chars_per_token": round(chars_per_token, 3),
    "training_time_minutes": round(elapsed / 60, 2),
}
with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)
print("saved to", args.out_dir)