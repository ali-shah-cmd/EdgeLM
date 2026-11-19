import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, SLM

# ---- settings ----
prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the capital of France?"
n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ckpt_path = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/sft_best.pt"

temperature = 0.7
top_p = 0.9
repetition_penalty = 1.0  # 1.0 = off; try 1.1-1.2 only if you see loops


def build_prompt(user_text):
    # MUST match the format in chat_format.py exactly (spaces/newlines).
    return f"User: {user_text}\nAssistant:"


# ---- load model ----
ckpt = torch.load(ckpt_path, map_location="cpu")
cfg = Config(**ckpt["cfg"])
model = SLM(cfg)
model.load_state_dict(ckpt["model"])
model.eval()
print(f"checkpoint {ckpt_path}: step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}\n")

# ---- tokenize prompt ----
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
eos_id = tok.token_to_id("<eos>")

ids = [eos_id] + tok.encode(build_prompt(prompt)).ids
x = torch.tensor([ids])
prompt_len = len(ids)

# ---- generate ----
with torch.no_grad():
    for _ in range(n_new):
        logits = model(x[:, -cfg.max_seq:])[:, -1, :]

        # 1. Repetition penalty (only on generated tokens, not the prompt)
        if repetition_penalty != 1.0:
            for token_id in set(x[0, prompt_len:].tolist()):
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
        sorted_logits[cumulative - probs > top_p] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)

        next_sorted = torch.multinomial(probs, num_samples=1)
        next_id = sorted_idx.gather(-1, next_sorted)
        x = torch.cat([x, next_id], dim=1)

        # 4. Stop when the model ends its answer
        if next_id.item() == eos_id:
            break

# ---- print only the model's answer, specials visible ----
answer_ids = x[0, prompt_len:].tolist()
print(tok.decode(answer_ids, skip_special_tokens=False))