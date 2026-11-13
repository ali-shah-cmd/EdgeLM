import argparse
import math
import os
import json
from datetime import datetime
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from model import Config, SLM

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--accum", type=int, default=16)
parser.add_argument("--seq", type=int, default=1024)
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--eval_every", type=int, default=250)
args = parser.parse_args()

device = "cuda"
torch.manual_seed(0)
np.random.seed(0)
os.makedirs("checkpoints", exist_ok=True)

os.makedirs("logs", exist_ok=True)

log_txt = "logs/training.log"
log_json = "logs/training.json"

# Initialize JSON training history
training_history = {
    "start_time": datetime.now().isoformat(),
    "config": {
        "steps": args.steps,
        "micro_bs": args.micro_bs,
        "accum": args.accum,
        "seq": args.seq,
        "lr": args.lr,
    },
    "metrics": []
}

def write_log(message):
    print(message)

    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def save_metrics():
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=4)

cfg = Config()
model = SLM(cfg).to(device)

train_data = np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")
val_data = np.memmap("data/tokenized/val.bin", dtype=np.uint16, mode="r")


def get_batch(data):
    ix = np.random.randint(0, len(data) - args.seq - 1, size=args.micro_bs)
    x = np.stack([data[i : i + args.seq] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + 1 + args.seq] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def compute_loss(x, y):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))


decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.dim() >= 2 and "emb" not in name and "table" not in name:
        decay.append(p)
    else:
        no_decay.append(p)

opt = torch.optim.AdamW(
    [
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=args.lr,
    betas=(0.9, 0.95),
    fused=True,
)

warmup = max(1, args.steps // 20)


def get_lr(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, args.steps - warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(n_batches=20):
    model.eval()
    total = 0.0
    for _ in range(n_batches):
        total += compute_loss(*get_batch(val_data)).item()
    model.train()
    return total / n_batches


def save(step, val_loss):
    torch.save(
        {"model": model.state_dict(), "cfg": asdict(cfg), "step": step, "val_loss": val_loss},
        "checkpoints/ckpt.pt",
    )


tokens_per_step = args.micro_bs * args.accum * args.seq
print(f"tokens per step: {tokens_per_step:,} | total tokens: {tokens_per_step * args.steps:,}")

model.train()
t0 = time.time()
for step in range(args.steps):
    lr = get_lr(step)
    for group in opt.param_groups:
        group["lr"] = lr

    step_loss = 0.0
    for _ in range(args.accum):
        x, y = get_batch(train_data)
        loss = compute_loss(x, y) / args.accum
        loss.backward()
        step_loss += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step % 10 == 0:
        speed = tokens_per_step * (step + 1) / (time.time() - t0)
        mem = torch.cuda.max_memory_allocated() / 1e9

        write_log(
            f"step {step + 1:5d} | "
            f"loss {step_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad_norm:.2f} | "
            f"{speed:,.0f} tok/s | "
            f"{mem:.2f} GB"
        )

        training_history["metrics"].append({
            "step": step + 1,
            "train_loss": step_loss,
            "learning_rate": lr,
            "grad_norm": grad_norm.item(),
            "tokens_per_second": speed,
            "gpu_memory_gb": mem,
            "val_loss": None
        })

        save_metrics()

    if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:

        val_loss = evaluate()

        write_log(
            f">>> step {step + 1} | val loss {val_loss:.4f}"
        )

        save(step + 1, val_loss)

        # Attach validation loss to the corresponding metric
        for metric in reversed(training_history["metrics"]):
            if metric["step"] == step + 1:
                metric["val_loss"] = val_loss
                break

        save_metrics()