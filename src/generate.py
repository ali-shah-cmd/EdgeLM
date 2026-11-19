import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, SLM

# ---------------- Settings ----------------
prompt = sys.argv[1] if len(sys.argv) > 1 else "The history of"
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 80

temperature = 0.4            # lower = more stable (good for arithmetic/logic)
top_p = 0.9
repetition_penalty = 1.05    # 1.0 disables it; 1.2 was too aggressive for short answers

# Fallback stop strings, in case the model doesn't emit <eos>.
# Adjust these to match your training format (e.g. "Q:", "###").
STOP_STRINGS = ["\n\n"]
# ------------------------------------------

ckpt = torch.load("checkpoints/reasoning_cont_latest.pt", map_location="cpu")
cfg = Config(**ckpt["cfg"])
model = SLM(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint from step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}\n")

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
eos_id = tok.token_to_id("<eos>")

ids = [eos_id] + tok.encode(prompt).ids
x = torch.tensor([ids])
prompt_len = x.shape[1]

with torch.no_grad():
    for _ in range(n_new):
        logits = model(x[:, -cfg.max_seq:])[:, -1, :]

        # 1. Repetition penalty (never applied to <eos>, so stopping stays possible)
        if repetition_penalty != 1.0:
            for token_id in set(x[0].tolist()):
                if token_id == eos_id:
                    continue
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        # 2. Temperature
        logits = logits / temperature

        # 3. Top-p (nucleus) sampling
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative - probs > top_p
        sorted_logits[remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)

        next_sorted = torch.multinomial(probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)

        # 4. Stop on <eos> (don't append it)
        if next_id.item() == eos_id:
            break

        x = torch.cat([x, next_id], dim=1)

        # 5. Fallback: stop on a stop string in the generated text
        generated = tok.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True)
        if any(s in generated for s in STOP_STRINGS):
            break

# Trim anything after the first stop string
generated = tok.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True)
for s in STOP_STRINGS:
    if s in generated:
        generated = generated.split(s)[0]

print(prompt + generated)