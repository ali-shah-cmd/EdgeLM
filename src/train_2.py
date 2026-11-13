
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

parser.add_argument("--steps", type=int, default=85000)
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--accum", type=int, default=16)
parser.add_argument("--seq", type=int, default=1024)
parser.add_argument("--lr", type=float, default=6e-4)

parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--save_every", type=int, default=500)
parser.add_argument("--warmup_steps", type=int, default=500)

parser.add_argument("--resume", type=str, default=None)

args = parser.parse_args()


# ==========================================
# 2. INITIALIZATION
# ==========================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

rng = np.random.default_rng(0)
val_rng = np.random.default_rng(1)

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_json = "logs/training.json"
log_file = "logs/training.log"

tokens_per_step = (
    args.micro_bs *
    args.accum *
    args.seq
)

total_tokens = tokens_per_step * args.steps


# ==========================================
# 3. TRAINING HISTORY AND LOGGING
# ==========================================

training_history = {
    "start_time": datetime.now().isoformat(),

    "config": {
        "steps": args.steps,
        "micro_bs": args.micro_bs,
        "accum": args.accum,
        "seq": args.seq,
        "lr": args.lr,
        "eval_every": args.eval_every,
        "save_every": args.save_every,
        "warmup_steps": args.warmup_steps,
    },

    "metrics": []
}


def save_metrics():

    temp_path = log_json + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    os.replace(temp_path, log_json)


def write_log(message):

    print(message, flush=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


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

if len(train_data) <= args.seq + 1:
    raise ValueError("Training dataset is too small.")

if len(val_data) <= args.seq + 1:
    raise ValueError("Validation dataset is too small.")


# ==========================================
# 6. BATCH GENERATION
# ==========================================

def get_batch(data, batch_rng):

    ix = batch_rng.integers(
        0,
        len(data) - args.seq - 1,
        size=args.micro_bs
    )

    x = np.stack([
        data[i:i + args.seq] for i in ix
    ]).astype(np.int64)

    y = np.stack([
        data[i + 1:i + 1 + args.seq] for i in ix
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

warmup = max(1, min(args.warmup_steps, args.steps))

min_lr_ratio = 0.1


def get_lr(step):

    # Warmup
    if step < warmup:

        return args.lr * (step + 1) / warmup

    # Cosine decay
    progress = (
        (step - warmup) /
        max(1, args.steps - warmup)
    )

    progress = min(1.0, max(0.0, progress))

    return args.lr * (
        min_lr_ratio +
        (1 - min_lr_ratio) * 0.5 *
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

        x, y = get_batch(val_data, val_rng)

        loss = compute_loss(x, y)

        total += loss.item()

    model.train()

    return total / n_batches


# ==========================================
# 11. CHECKPOINT STATE
# ==========================================

start_step = 0

best_val_loss = float("inf")

last_val_loss = None


# ==========================================
# 12. CHECKPOINT SAVING
# ==========================================

def save_checkpoint(path, step):

    torch.cuda.synchronize()

    checkpoint = {

        # Model
        "model": model.state_dict(),

        # Optimizer
        "optimizer": opt.state_dict(),

        # Configuration
        "cfg": asdict(cfg),

        # Training progress
        "step": step,

        # Validation
        "val_loss": last_val_loss,
        "best_val_loss": best_val_loss,

        # Random states
        "train_rng_state": rng.bit_generator.state,
        "val_rng_state": val_rng.bit_generator.state,

        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),

        # Training history
        "training_history": training_history,
    }

    temp_path = path + ".tmp"

    # Write temporary checkpoint first.
    torch.save(checkpoint, temp_path)

    # Replace previous checkpoint.
    os.replace(temp_path, path)

    write_log(f"Checkpoint saved: {path}")


# ==========================================
# 13. RESUME TRAINING
# ==========================================

if args.resume is not None:

    write_log(f"Loading checkpoint: {args.resume}")

    checkpoint = torch.load(
        args.resume,
        map_location=device,
        weights_only=False
    )

    required_keys = [
        "model",
        "optimizer",
        "step",
        "train_rng_state",
        "val_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise RuntimeError(
            f"Checkpoint cannot resume training. "
            f"Missing: {missing_keys}"
        )

    model.load_state_dict(checkpoint["model"])

    opt.load_state_dict(checkpoint["optimizer"])

    start_step = checkpoint["step"]

    best_val_loss = checkpoint.get(
        "best_val_loss",
        float("inf")
    )

    last_val_loss = checkpoint.get("val_loss")

    rng.bit_generator.state = checkpoint["train_rng_state"]

    val_rng.bit_generator.state = checkpoint["val_rng_state"]

    torch.set_rng_state(
        checkpoint["torch_rng_state"].cpu()
    )

    torch.cuda.set_rng_state_all(
        [
            state.cpu()
            for state in checkpoint["cuda_rng_state"]
        ]
    )

    training_history = checkpoint.get(
        "training_history",
        training_history
    )

    write_log(f"Resuming from step {start_step}")

    if start_step >= args.steps:
        raise ValueError(
            "Checkpoint has already reached the requested "
            "training step count."
        )


# ==========================================
# 14. TRAINING INFORMATION
# ==========================================

print("\n========== TRAINING CONFIGURATION ==========")

print(f"Total training steps: {args.steps:,}")
print(f"Starting step: {start_step:,}")

print(f"Tokens per step: {tokens_per_step:,}")
print(f"Planned total tokens: {total_tokens:,}")

print(f"Learning rate: {args.lr}")
print(f"Warmup steps: {warmup}")

print(f"Validation interval: {args.eval_every}")
print(f"Checkpoint interval: {args.save_every}")

print("============================================\n")


# ==========================================
# 15. TRAINING LOOP
# ==========================================

model.train()

torch.cuda.reset_peak_memory_stats()

t0 = time.time()

for step in range(start_step, args.steps):

    # --------------------------------------
    # Learning rate
    # --------------------------------------

    lr = get_lr(step)

    for group in opt.param_groups:
        group["lr"] = lr


    # --------------------------------------
    # Gradient accumulation
    # --------------------------------------

    opt.zero_grad(set_to_none=True)

    step_loss = 0.0

    for _ in range(args.accum):

        x, y = get_batch(train_data, rng)

        loss = compute_loss(x, y) / args.accum

        loss.backward()

        step_loss += loss.detach()


    # --------------------------------------
    # Gradient clipping
    # --------------------------------------

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )


    # --------------------------------------
    # Optimizer update
    # --------------------------------------

    opt.step()

    global_step = step + 1


    # --------------------------------------
    # Progress logging
    # --------------------------------------

    should_log = (
        global_step % 10 == 0
        or global_step == args.steps
    )

    if should_log:

        elapsed = time.time() - t0

        completed_steps = global_step - start_step

        speed = (
            tokens_per_step * completed_steps /
            max(elapsed, 1e-8)
        )

        mem = (
            torch.cuda.max_memory_allocated() /
            1e9
        )

        train_loss = step_loss.item()

        grad_value = grad_norm.item()

        message = (
            f"step {global_step:6d} | "
            f"loss {train_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad_value:.2f} | "
            f"{speed:,.0f} tok/s | "
            f"{mem:.2f} GB"
        )

        write_log(message)

        training_history["metrics"].append({

            "step": global_step,

            "train_loss": train_loss,

            "learning_rate": lr,

            "grad_norm": grad_value,

            "tokens_per_second": speed,

            "gpu_memory_gb": mem,

            "val_loss": None,
        })


    # --------------------------------------
    # Validation
    # --------------------------------------

    should_evaluate = (
        global_step % args.eval_every == 0
        or global_step == args.steps
    )

    if should_evaluate:

        last_val_loss = evaluate()

        message = (
            f">>> step {global_step} | "
            f"validation loss {last_val_loss:.4f}"
        )

        write_log(message)

        # Update validation metric.
        if (
            training_history["metrics"]
            and training_history["metrics"][-1]["step"] == global_step
        ):

            training_history["metrics"][-1]["val_loss"] = last_val_loss


        # ----------------------------------
        # Save BEST checkpoint
        # ----------------------------------

        if last_val_loss < best_val_loss:

            best_val_loss = last_val_loss

            save_checkpoint(
                "checkpoints/best.pt",
                global_step
            )

            write_log(
                f"New best validation loss: {best_val_loss:.4f}"
            )


    # --------------------------------------
    # Save LATEST checkpoint
    # --------------------------------------

    should_save = (
        global_step % args.save_every == 0
        or global_step == args.steps
    )

    if should_save:

        save_checkpoint(
            "checkpoints/latest.pt",
            global_step
        )


    # --------------------------------------
    # Save metrics
    # --------------------------------------

    if should_evaluate or should_save:

        save_metrics()


# ==========================================
# 16. TRAINING COMPLETE
# ==========================================

training_history["end_time"] = datetime.now().isoformat()

save_metrics()

write_log("\nTraining completed successfully.")

write_log("Latest checkpoint: checkpoints/latest.pt")

write_log("Best checkpoint: checkpoints/best.pt")

write_log("Training log: logs/training.log")

write_log("Training metrics: logs/training.json")
