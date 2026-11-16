import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, SLM

# ---------- arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/latest.pt")
parser.add_argument("--name", default=None)  # default: built from the real step
parser.add_argument("--n_batches", type=int, default=200)
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--seq", type=int, default=1024)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- load model ----------
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = Config(**ckpt["cfg"])
model = SLM(cfg)
model.load_state_dict(ckpt["model"])
model.to(device).eval()

step = ckpt["step"]
name = args.name or f"step{step}"
print(f"loaded {args.ckpt} | trained step {step} | "
      f"val loss recorded by train.py: {ckpt.get('val_loss')}")

tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = tok.token_to_id("<eos>")

# ---------- part A: fixed validation loss ----------
val = np.memmap("data/tokenized/val.bin", dtype=np.uint16, mode="r")

n_windows = args.n_batches * args.micro_bs
starts = np.linspace(0, len(val) - args.seq - 1, n_windows).astype(np.int64)

total = 0.0
with torch.no_grad():
    for b in range(args.n_batches):
        idx = starts[b * args.micro_bs:(b + 1) * args.micro_bs]
        x = np.stack([val[i:i + args.seq] for i in idx]).astype(np.int64)
        y = np.stack([val[i + 1:i + 1 + args.seq] for i in idx]).astype(np.int64)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))
        total += loss.item()

val_loss = total / args.n_batches
header = (f"[{name}] step {step} | fixed val loss: {val_loss:.4f} "
          f"(perplexity {np.exp(val_loss):.2f}, {n_windows * args.seq:,} tokens)")
print(header)

# ---------- part B: fixed prompts ----------
PROMPTS = [
    "The capital of Pakistan is",
    "Photosynthesis is the process by which",
    "Machine learning is",
    "Water boils at a temperature of",
    "In 1947, the subcontinent",
    "2+2=",
    "12 + 30 =",
    "Question: What is the capital of France?\nAnswer:",
    "User: find a red ball\nAssistant:",
    "Once upon a time, a small robot",
]

TEMPERATURE, TOP_P, REP_PENALTY, REP_WINDOW, N_NEW = 0.8, 0.9, 1.1, 64, 70


@torch.no_grad()
def sample(prompt):
    ids = [EOS] + tok.encode(prompt).ids
    x = torch.tensor([ids], device=device)
    for _ in range(N_NEW):
        logits = model(x[:, -cfg.max_seq:])[:, -1, :].float()

        for t in set(x[0, -REP_WINDOW:].tolist()):
            logits[0, t] = logits[0, t] / REP_PENALTY if logits[0, t] > 0 else logits[0, t] * REP_PENALTY

        logits = logits / TEMPERATURE
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = torch.cumsum(probs, dim=-1) - probs > TOP_P
        sorted_logits[remove] = float("-inf")
        probs = F.softmax(sorted_logits, dim=-1)
        nxt = sorted_idx.gather(-1, torch.multinomial(probs, 1))
        x = torch.cat([x, nxt], dim=1)
    return tok.decode(x[0, 1:].tolist(), skip_special_tokens=False)


os.makedirs("eval_results", exist_ok=True)
lines = [header, ""]
for p in PROMPTS:
    torch.manual_seed(0)  # same randomness for every prompt and every run
    out = sample(p)
    block = f"### {p!r}\n{out}\n"
    print(block)
    lines.append(block)

with open(f"eval_results/{name}.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"saved eval_results/{name}.txt")