import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
eos = tok.token_to_id("<eos>")

d = np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")
print("total tokens:", len(d), "| eos id:", eos)
print("first 8 ids:", d[:8].tolist())
print("eos positions in first 5000 tokens:", np.nonzero(d[:5000] == eos)[0][:10].tolist())
print()
print(tok.decode(d[:60].tolist(), skip_special_tokens=False))