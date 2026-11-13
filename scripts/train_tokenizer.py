import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

SPECIALS = ["<pad>", "<bos>", "<eos>", "<think>", "</think>"]

def texts():
    with open("data/raw/fineweb_edu.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size = 16384,
    special_tokens = SPECIALS,
    initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tok.train_from_iterator(texts(), trainer)
tok.save("tokenizer/tokenizer.json")

print("vocab size:", tok.get_vocab_size())
print("special ids:", [tok.token_to_id(s) for s in SPECIALS])

sample = "The cat sat on the mat. <think>2 + 2 = 4</think>"
enc = tok.encode(sample)
print("tokens:", enc.tokens)
print("ids   :", enc.ids)
print("round trip ok:", tok.decode(enc.ids, skip_special_tokens=False) == sample)