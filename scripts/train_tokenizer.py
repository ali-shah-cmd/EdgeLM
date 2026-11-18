
import os
import json
import time

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/raw/fineweb_edu.jsonl"
TOKENIZER_PATH = "tokenizer/tokenizer.json"

VOCAB_SIZE = 16384

SPECIALS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<think>",
    "</think>"
]


# ==========================================
# 2. DIRECTORY SETUP
# ==========================================

os.makedirs("tokenizer", exist_ok=True)


# ==========================================
# 3. TEXT GENERATOR
# ==========================================

def texts():

    with open(DATA_PATH, "r", encoding="utf-8") as f:

        for line in f:

            if not line.strip():
                continue

            try:

                document = json.loads(line)

                text = document.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                yield text

            except json.JSONDecodeError:
                continue


# ==========================================
# 4. TOKENIZER INITIALIZATION
# ==========================================

tok = Tokenizer(models.BPE())

tok.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

tok.decoder = decoders.ByteLevel()


# ==========================================
# 5. TRAINER CONFIGURATION
# ==========================================

trainer = trainers.BpeTrainer(

    vocab_size=VOCAB_SIZE,

    special_tokens=SPECIALS,

    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),

    show_progress=True,

    min_frequency=2
)


# ==========================================
# 6. TRAIN TOKENIZER
# ==========================================

print("Starting tokenizer training...")

start_time = time.time()

tok.train_from_iterator(
    texts(),
    trainer=trainer
)

elapsed = time.time() - start_time


# ==========================================
# 7. SAVE TOKENIZER
# ==========================================

tok.save(TOKENIZER_PATH)

print("\nTokenizer training completed.")

print(f"Training time: {elapsed / 60:.2f} minutes")

print(f"Vocabulary size: {tok.get_vocab_size()}")

print("\nSpecial token IDs:")

for token in SPECIALS:

    print(token, ":", tok.token_to_id(token))


# ==========================================
# 8. TOKENIZER TEST
# ==========================================

sample = "The cat sat on the mat. <think>2 + 2 = 4</think>"

enc = tok.encode(sample)

decoded = tok.decode(
    enc.ids,
    skip_special_tokens=False
)

print("\nOriginal text:")
print(sample)

print("\nTokens:")
print(enc.tokens)

print("\nToken IDs:")
print(enc.ids)

print("\nDecoded text:")
print(decoded)

print("\nRound trip successful:", decoded == sample)


# ==========================================
# 9. SAVE TOKENIZER INFORMATION
# ==========================================

metadata = {

    "vocab_size": tok.get_vocab_size(),

    "special_tokens": {

        token: tok.token_to_id(token)

        for token in SPECIALS
    },

    "algorithm": "ByteLevel BPE",

    "training_time_minutes": round(elapsed / 60, 2)

}

with open(
    "tokenizer/metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(metadata, f, indent=4)

print("\nTokenizer metadata saved.")