import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, SLM

prompt = sys.argv[1] if len(sys.argv) > 1 else "The history of"
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 80
temperature = 0.8
top_k = 40

ckpt = torch.load("checkpoints/Ckpt/best.pt", map_location="cpu")
cfg = Config(**ckpt["cfg"])
model = SLM(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint from step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}\n")

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
ids = [tok.token_to_id("<eos>")] + tok.encode(prompt).ids
x = torch.tensor([ids])

# with torch.no_grad():
#     for _ in range(n_new):
#         logits = model(x[:, -cfg.max_seq:])[:, -1, :] / temperature
#         top_values, _ = torch.topk(logits, top_k)
#         logits[logits < top_values[:, -1:]] = float("-inf")
#         probs = F.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1)
#         x = torch.cat([x, next_id], dim=1)

# print(tok.decode(x[0, 1:].tolist(), skip_special_tokens=False))

eos_id = tok.token_to_id("<eos>")

# with torch.no_grad():
#     for _ in range(n_new):
#         logits = model(x[:, -cfg.max_seq:])[:, -1, :] / temperature

#         # Apply top-k filtering only when top_k is greater than zero.
#         if top_k > 0:
#             top_values, _ = torch.topk(logits, top_k)
#             logits[logits < top_values[:, -1:]] = float("-inf")

#         probs = F.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1)

#         x = torch.cat([x, next_id], dim=1)

#         if next_id.item() == eos_id:
#             break

repetition_penalty = 1.2
top_p = 0.9

with torch.no_grad():
    for _ in range(n_new):
        logits = model(x[:, -cfg.max_seq:])[:, -1, :]

        # 1. Penalize tokens already generated
        for token_id in set(x[0].tolist()):
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
        remove = cumulative - probs > top_p      # tokens beyond the top-p mass
        sorted_logits[remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)

        next_sorted = torch.multinomial(probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)
        x = torch.cat([x, next_id], dim=1)

print(tok.decode(
    x[0, 1:].tolist(),
    skip_special_tokens=True
))