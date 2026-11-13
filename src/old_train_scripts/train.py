import argparse
import math
import os
import json
import time

from datetime import datetime
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from model import Config, SLM


# ==========================================
# 1. ARGUMENTS
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--accum", type=int, default=16)
parser.add_argument("--seq", type=int, default=1024)
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--eval_every", type=int, default=250)

args = parser.parse_args()


# ==========================================
# 2. INITIALIZATION
# ==========================================

device = "cuda"

torch.manual_seed(0)

# Modern NumPy random generator.
# Supports large integer ranges on Windows.
rng = np.random.default_rng(0)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_txt = "logs/training.log"
log_json = "logs/training.json"

# Start fresh log files for this run.
open(log_txt, "w", encoding="utf-8").close()

training_history = {
    "start_time": datetime.now().isoformat(),

    "config": {
        "steps": args.steps,
        "micro_bs": args.micro_bs,
        "accum": args.accum,
        "seq": args.seq,
        "lr": args.lr,
        "eval_every": args.eval_every,
    },

    "metrics": []
}


# ==========================================
# 3. LOGGING
# ==========================================

def write_log(message):

    print(message)

    with open(log_txt, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_metrics():

    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=4)


def update_metric(step, **values):

    for metric in training_history["metrics"]:

        if metric["step"] == step:
            metric.update(values)
            save_metrics()
            return

    metric = {
        "step": step,
        "train_loss": None,
        "learning_rate": None,
        "grad_norm": None,
        "tokens_per_second": None,
        "gpu_memory_gb": None,
        "val_loss": None,
    }

    metric.update(values)

    training_history["metrics"].append(metric)

    save_metrics()


# Save initial history immediately.
save_metrics()


# ==========================================
# 4. MODEL
# ==========================================

cfg = Config()

model = SLM(cfg).to(device)


# ==========================================
# 5. DATASET
# ==========================================

train_data = np.memmap(
    "data/tokenized/train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    "data/tokenized/val.bin",
    dtype=np.uint16,
    mode="r"
)


# ==========================================
# 6. BATCH GENERATION
# ==========================================

def get_batch(data):

    ix = rng.integers(
        0,
        len(data) - args.seq - 1,
        size=args.micro_bs
    )

    x = np.stack([
        data[i : i + args.seq] for i in ix
    ]).astype(np.int64)

    y = np.stack([
        data[i + 1 : i + 1 + args.seq] for i in ix
    ]).astype(np.int64)

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y


# ==========================================
# 7. LOSS
# ==========================================

def compute_loss(x, y):

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16
    ):

        logits = model(x)

    loss = F.cross_entropy(
        logits.float().view(-1, cfg.vocab_size),
        y.reshape(-1)
    )

    return loss


# ==========================================
# 8. OPTIMIZER
# ==========================================

decay = []
no_decay = []

for name, p in model.named_parameters():

    if p.dim() >= 2 and "emb" not in name and "table" not in name:
        decay.append(p)

    else:
        no_decay.append(p)


opt = torch.optim.AdamW(
    [
        {
            "params": decay,
            "weight_decay": 0.1
        },
        {
            "params": no_decay,
            "weight_decay": 0.0
        }
    ],

    lr=args.lr,

    betas=(0.9, 0.95),

    fused=True
)


# ==========================================
# 9. LEARNING RATE SCHEDULER
# ==========================================

warmup = max(1, args.steps // 20)


def get_lr(step):

    if step < warmup:

        return args.lr * (step + 1) / warmup

    progress = (
        (step - warmup) /
        max(1, args.steps - warmup)
    )

    return args.lr * (
        0.1 + 0.9 * 0.5 *
        (1 + math.cos(math.pi * progress))
    )


# ==========================================
# 10. VALIDATION
# ==========================================

@torch.no_grad()
def evaluate(n_batches=20):

    model.eval()

    total = 0.0

    for _ in range(n_batches):

        x, y = get_batch(val_data)

        loss = compute_loss(x, y)

        total += loss.item()

    model.train()

    return total / n_batches


# ==========================================
# 11. CHECKPOINT SAVING
# ==========================================

def save(step, val_loss):

    torch.save(
        {
            "model": model.state_dict(),
            "cfg": asdict(cfg),
            "step": step,
            "val_loss": val_loss,
        },

        "checkpoints/ckpt.pt"
    )


# ==========================================
# 12. TRAINING INFORMATION
# ==========================================

tokens_per_step = (
    args.micro_bs *
    args.accum *
    args.seq
)

total_tokens = tokens_per_step * args.steps

write_log(
    f"tokens per step: {tokens_per_step:,} | "
    f"total tokens: {total_tokens:,}"
)


# ==========================================
# 13. TRAINING LOOP
# ==========================================

model.train()

t0 = time.time()

for step in range(args.steps):

    # Current learning rate
    lr = get_lr(step)

    for group in opt.param_groups:
        group["lr"] = lr


    # --------------------------------------
    # Gradient accumulation
    # --------------------------------------

    step_loss = 0.0

    for _ in range(args.accum):

        x, y = get_batch(train_data)

        loss = compute_loss(x, y) / args.accum

        loss.backward()

        step_loss += loss.item()


    # --------------------------------------
    # Gradient clipping and optimizer update
    # --------------------------------------

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    opt.step()

    opt.zero_grad(set_to_none=True)


    global_step = step + 1


    # --------------------------------------
    # Progress logging
    # --------------------------------------

    should_log = (
        global_step % 10 == 0
        or global_step % args.eval_every == 0
        or global_step == args.steps
    )

    if should_log:

        speed = (
            tokens_per_step * global_step /
            (time.time() - t0)
        )

        mem = (
            torch.cuda.max_memory_allocated() /
            1e9
        )

        write_log(
            f"step {global_step:5d} | "
            f"loss {step_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad_norm.item():.2f} | "
            f"{speed:,.0f} tok/s | "
            f"{mem:.2f} GB"
        )

        update_metric(
            global_step,

            train_loss=step_loss,

            learning_rate=lr,

            grad_norm=grad_norm.item(),

            tokens_per_second=speed,

            gpu_memory_gb=mem
        )


    # --------------------------------------
    # Validation and checkpoint saving
    # --------------------------------------

    if (
        global_step % args.eval_every == 0
        or global_step == args.steps
    ):

        val_loss = evaluate()

        write_log(
            f">>> step {global_step} | "
            f"val loss {val_loss:.4f}"
        )

        # Save checkpoint
        save(global_step, val_loss)

        # Record validation loss in JSON
        update_metric(
            global_step,
            val_loss=val_loss
        )


# ==========================================
# 14. TRAINING COMPLETE
# ==========================================

training_history["end_time"] = datetime.now().isoformat()

save_metrics()

write_log("\nTraining completed successfully.")

write_log("Checkpoint saved: checkpoints/ckpt.pt")

write_log("Training logs saved:")
write_log(" - logs/training.log")
write_log(" - logs/training.json")
