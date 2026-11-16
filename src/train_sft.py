import argparse
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, SLM

# ---------- arguments ----------
parser = argparse.ArgumentParser()
parser.add_argument("--base", default="checkpoints/base_final.pt")
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--max_steps", type=int, default=0)  # >0: short test run
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--accum", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--warmup", type=int, default=100)
parser.add_argument("--eval_every", type=int, default=1000)
parser.add_argument("--save_every", type=int, default=1000)
parser.add_argument("--resume", default=None)
args = parser.parse_args()

device = "cuda"
torch.manual_seed(0)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def log(msg):
    print(msg, flush=True)
    with open("logs/sft.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


tok = Tokenizer.from_file("tokenizer/tokenizer.json")
EOS = tok.token_to_id("<eos>")
PAD = tok.token_to_id("<pad>")

# ---------- data ----------
ids_all = np.load("data/sft/ids.npy", mmap_mode="r")
mask_all = np.load("data/sft/mask.npy", mmap_mode="r")

perm = np.random.default_rng(0).permutation(len(ids_all))
val_rows = np.sort(perm[:1000])       # 1000 conversations held out
train_rows = perm[1000:]

rows_per_step = args.micro_bs * args.accum
steps_per_epoch = len(train_rows) // rows_per_step
total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs

# ---------- model ----------
base = torch.load(args.base, map_location="cpu", weights_only=False)
cfg = Config(**base["cfg"])
model = SLM(cfg)
model.load_state_dict(base["model"])
model.to(device)
del base

decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.dim() >= 2 and "emb" not in name and "table" not in name:
        decay.append(p)
    else:
        no_decay.append(p)

opt = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.01},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), fused=True,
)


def get_lr(step):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = (step - args.warmup) / max(1, total_steps - args.warmup)
    progress = min(1.0, progress)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------- batches and loss ----------
def get_rows(row_ids):
    row_ids = np.sort(row_ids)
    ids = torch.from_numpy(ids_all[row_ids].astype(np.int64))
    mask = torch.from_numpy(mask_all[row_ids].astype(np.bool_))
    T = int((ids != PAD).sum(1).max())          # longest real row in this batch
    ids, mask = ids[:, :T], mask[:, :T]
    x = ids[:, :-1]
    y = ids[:, 1:].clone()
    y[~mask[:, 1:]] = -100                      # no loss on prompt or padding
    return x.to(device), y.to(device)


def batch_loss_sum(x, y):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1),
        ignore_index=-100, reduction="sum",
    )


@torch.no_grad()
def evaluate(n_eval_rows=400):
    model.eval()
    total, count = 0.0, 0
    rows = val_rows[:n_eval_rows]
    for i in range(0, len(rows), args.micro_bs):
        x, y = get_rows(rows[i:i + args.micro_bs])
        total += batch_loss_sum(x, y).item()
        count += (y != -100).sum().item()
    model.train()
    return total / count


@torch.no_grad()
def sample_reply(user_text, n_new=80):
    model.eval()
    ids = [EOS] + tok.encode(f"User: {user_text}\nAssistant:").ids
    x = torch.tensor([ids], device=device)
    for _ in range(n_new):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x[:, -cfg.max_seq:])[:, -1, :]
        nxt = logits.argmax(-1, keepdim=True)
        if nxt.item() == EOS:
            break
        x = torch.cat([x, nxt], dim=1)
    model.train()
    return tok.decode(x[0, len(ids):].tolist(), skip_special_tokens=False)


best_val = float("inf")


def save_ckpt(path, step, val_loss):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "cfg": asdict(cfg), "step": step, "val_loss": val_loss,
                "best_val": best_val}, tmp)
    os.replace(tmp, path)
    log(f"checkpoint saved: {path}")


start_step = 0
if args.resume:
    ck = torch.load(args.resume, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    start_step, best_val = ck["step"], ck["best_val"]
    log(f"resumed from {args.resume} at step {start_step}")

log(f"train rows {len(train_rows):,} | steps/epoch {steps_per_epoch:,} | "
    f"total steps {total_steps:,} | rows/step {rows_per_step}")
last_val = evaluate()
log(f">>> step {start_step} | val loss {last_val:.4f}  (assistant tokens only)")

# ---------- training loop ----------
model.train()
order, order_epoch = None, -1
t0 = time.time()

for step in range(start_step, total_steps):
    epoch, pos = divmod(step, steps_per_epoch)
    if epoch != order_epoch:
        order = np.random.default_rng(1000 + epoch).permutation(train_rows)
        order_epoch = epoch
    step_rows = order[pos * rows_per_step:(pos + 1) * rows_per_step]

    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr

    micro = [get_rows(step_rows[i:i + args.micro_bs])
             for i in range(0, rows_per_step, args.micro_bs)]
    n_tok = sum((y != -100).sum().item() for _, y in micro)

    opt.zero_grad(set_to_none=True)
    step_loss = 0.0
    for x, y in micro:
        loss = batch_loss_sum(x, y) / n_tok      # mean over ALL trained tokens of the step
        loss.backward()
        step_loss += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    gs = step + 1

    if gs % 10 == 0 or gs == total_steps:
        done = gs - start_step
        eta_h = (time.time() - t0) / done * (total_steps - gs) / 3600
        log(f"step {gs:6d} | loss {step_loss:.4f} | lr {lr:.2e} | "
            f"grad {grad_norm.item():.2f} | eta {eta_h:.1f} h")

    if gs % args.eval_every == 0 or gs == total_steps:
        last_val = evaluate()
        log(f">>> step {gs} | val loss {last_val:.4f}")
        log("sample: " + repr(sample_reply("What is the capital of France?")))
        if last_val < best_val:
            best_val = last_val
            save_ckpt("checkpoints/sft_best.pt", gs, last_val)

    if gs % args.save_every == 0 or gs == total_steps:
        save_ckpt("checkpoints/sft_latest.pt", gs, last_val)