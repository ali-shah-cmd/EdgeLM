import argparse
import json
import math

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from model import Config, SLM

# ---------- arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default="checkpoints/base_final.pt")
parser.add_argument("--others", nargs="*", default=["HuggingFaceTB/SmolLM2-135M"])
parser.add_argument("--n_docs", type=int, default=300)
parser.add_argument("--chars", type=int, default=2500)  # keeps every doc under 1024 tokens
parser.add_argument("--device", default="cpu")  # CPU so a running training job is not disturbed
args = parser.parse_args()

# ---------- same raw text for every model ----------
# These are the documents your prep_data.py put in the validation split (doc_id % 100 == 0).
docs = []
with open("data/raw/fineweb_edu.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i % 100 == 0:
            docs.append(json.loads(line)["text"][:args.chars])
            if len(docs) == args.n_docs:
                break

total_bytes = sum(len(t.encode("utf-8")) for t in docs)
print(f"{len(docs)} documents, {total_bytes:,} bytes of text\n")


def report(label, nll_nats, n_tokens):
    bpb = nll_nats / (math.log(2) * total_bytes)
    print(f"{label:34s} tokens {n_tokens:8,d} | loss/token {nll_nats / n_tokens:.4f} | bits/byte {bpb:.4f}")
    return bpb


def doc_loss(logits, y):
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), reduction="sum").item()


# ---------- your model ----------
ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = Config(**ckpt["cfg"])
own = SLM(cfg)
own.load_state_dict(ckpt["model"])
own.to(args.device).eval()
own_tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = own_tok.token_to_id("<eos>")

nll, n = 0.0, 0
with torch.no_grad():
    for t in docs:
        ids = [EOS] + own_tok.encode(t).ids
        assert len(ids) <= 1025, "document too long for 1024 tokens: lower --chars"
        x = torch.tensor([ids[:-1]], device=args.device)
        y = torch.tensor([ids[1:]], device=args.device)
        nll += doc_loss(own(x), y)
        n += y.numel()
results = {f"yours (step {ckpt['step']})": report(f"yours (step {ckpt['step']})", nll, n)}
del own

# ---------- other models (SmolLM2 ...) ----------
for name in args.others:
    hf_tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(args.device).eval()
    sep = hf_tok.convert_tokens_to_ids("<|endoftext|>")  # document separator

    nll, n = 0.0, 0
    with torch.no_grad():
        for t in docs:
            ids = [sep] + hf_tok(t, add_special_tokens=False)["input_ids"]
            assert len(ids) <= 1025, "document too long for 1024 tokens: lower --chars"
            x = torch.tensor([ids[:-1]], device=args.device)
            y = torch.tensor([ids[1:]], device=args.device)
            nll += doc_loss(hf(x).logits, y)
            n += y.numel()
    results[name] = report(name, nll, n)
    del hf

# ---------- summary ----------
print()
mine = results[f"yours (step {ckpt['step']})"]
for name in args.others:
    print(f"gap to {name}: {mine - results[name]:+.4f} bits/byte "
          f"({(mine / results[name] - 1) * 100:+.1f}% worse)")