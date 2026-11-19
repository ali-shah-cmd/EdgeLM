import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
eos_id = tok.token_to_id("<eos>")

data = np.fromfile("data/tokenized/reasoning_synth.bin", dtype=np.uint16)  # adjust path and dtype
print("eos id:", eos_id)
print("eos count:", int((data == eos_id).sum()), "of", len(data))
print(tok.decode(data[:300].tolist(), skip_special_tokens=False))