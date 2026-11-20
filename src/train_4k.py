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

from model_4k import Config, SLM


# ============================================================
# 1. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Train from scratch on FineWeb 10BT mixed with reasoning data."
)

parser.add_argument("--steps", type=int, default=166000)
parser.add_argument("--micro_bs", type=int, default=1)
parser.add_argument("--accum", type=int, default=16)
parser.add_argument("--seq", type=int, default=4096)

parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--warmup_steps", type=int, default=500)

parser.add_argument("--eval_every", type=int, default=500)
parser.add_argument("--save_every", type=int, default=500)

parser.add_argument(
    "--unseen",
    type=str,
    default="data/tokenized/all_blocks_4096.npy",
    help="FineWeb block indices to read once each (scripts/make_block.py), or 'none' for random sampling."
)

parser.add_argument("--seed", type=int, default=1)

parser.add_argument(
    "--fill_seen",
    action="store_true",
    help="Use previously seen FineWeb blocks if unseen blocks run out."
)

parser.add_argument(
    "--tag",
    type=str,
    default="scratch4k",
    help="Prefix for this run's checkpoints and logs."
)

parser.add_argument(
    "--train_bin",
    type=str,
    default="data/tokenized/train.bin"
)

parser.add_argument(
    "--val_bin",
    type=str,
    default="data/tokenized/val.bin"
)

# New names are reasoning_pct and reasoning_bin.
# Old math_pct and math_bin names remain accepted for compatibility.

parser.add_argument(
    "--reasoning_pct",
    "--math_pct",
    dest="reasoning_pct",
    type=int,
    default=3,
    help="Percentage of training sequences sampled from reasoning data."
)

parser.add_argument(
    "--reasoning_bin",
    "--math_bin",
    dest="reasoning_bin",
    type=str,
    default="data/tokenized/reasoning_synth.bin",
    help="Combined tokenized reasoning dataset."
)

parser.add_argument(
    "--reasoning_val_tokens",
    type=int,
    default=500_000,
    help="Approximate number of final reasoning tokens reserved for validation."
)

parser.add_argument(
    "--reasoning_eos_id",
    type=int,
    default=2,
    help="EOS token ID used to split reasoning examples."
)

parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Resume this run from its own checkpoint."
)

args = parser.parse_args()


# ============================================================
# 2. ARGUMENT VALIDATION
# ============================================================

if args.steps <= 0:
    raise ValueError("--steps must be positive.")

if args.micro_bs <= 0 or args.accum <= 0 or args.seq <= 0:
    raise ValueError("Batch sizes and sequence length must be positive.")

if args.lr <= 0:
    raise ValueError("--lr must be positive.")

if args.warmup_steps < 0:
    raise ValueError("--warmup_steps cannot be negative.")

if args.eval_every <= 0 or args.save_every <= 0:
    raise ValueError("Evaluation and save intervals must be positive.")

if not 0 <= args.reasoning_pct < 100:
    raise ValueError("--reasoning_pct must be between 0 and 99.")

if args.reasoning_val_tokens <= args.seq + 1:
    raise ValueError("--reasoning_val_tokens must exceed seq + 1.")

if args.reasoning_eos_id < 0 or args.reasoning_eos_id >= 65536:
    raise ValueError("Invalid EOS token ID for uint16 data.")


# ============================================================
# 3. DEVICE AND RANDOM SEEDS
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = "cuda"

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

rng = np.random.default_rng(args.seed)
val_rng = np.random.default_rng(1)

# Prefer BF16 where supported.
# Otherwise use full precision rather than risk FP16 overflow
# without a gradient scaler.

use_bf16 = torch.cuda.is_bf16_supported()

if use_bf16:
    amp_enabled = True
    amp_dtype = torch.bfloat16
    amp_name = "bfloat16"
else:
    amp_enabled = False
    amp_dtype = torch.float32
    amp_name = "float32"


# ============================================================
# 4. OUTPUT PATHS AND LOGGING
# ============================================================

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

latest_path = f"checkpoints/{args.tag}_latest.pt"

best_fineweb_path = f"checkpoints/{args.tag}_best_fineweb.pt"
best_reasoning_path = f"checkpoints/{args.tag}_best_reasoning.pt"

log_json = f"logs/{args.tag}.json"
log_file = f"logs/{args.tag}.log"

if args.resume is None and os.path.exists(latest_path):
    raise RuntimeError(
        f"{latest_path} already exists.\n"
        f"Use --resume {latest_path} to resume it, "
        f"or choose another --tag."
    )

tokens_per_step = args.micro_bs * args.accum * args.seq
total_tokens = tokens_per_step * args.steps


# ============================================================
# 5. CONFIGURATION RECORD
# ============================================================

run_config = {
    "steps": args.steps,
    "micro_bs": args.micro_bs,
    "accum": args.accum,
    "seq": args.seq,
    "lr": args.lr,
    "warmup_steps": args.warmup_steps,
    "eval_every": args.eval_every,
    "save_every": args.save_every,
    "unseen": args.unseen,
    "seed": args.seed,
    "fill_seen": args.fill_seen,
    "tag": args.tag,
    "reasoning_pct": args.reasoning_pct,
    "reasoning_bin": args.reasoning_bin,
    "reasoning_val_tokens": args.reasoning_val_tokens,
    "reasoning_eos_id": args.reasoning_eos_id,
    "train_bin": args.train_bin,
    "val_bin": args.val_bin,
}


training_history = {
    "start_time": datetime.now().isoformat(),
    "config": run_config,
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


# ============================================================
# 6. MODEL
# ============================================================

cfg = Config()

if args.seq > cfg.max_seq:
    raise ValueError(
        f"--seq {args.seq} is larger than Config.max_seq {cfg.max_seq}. "
        "Raise max_seq in model.py."
    )

model = SLM(cfg).to(device)

n_params = sum(p.numel() for p in model.parameters())


# ============================================================
# 7. DATASETS
# ============================================================

if not os.path.exists(args.train_bin):
    raise FileNotFoundError(args.train_bin)

if not os.path.exists(args.val_bin):
    raise FileNotFoundError(args.val_bin)

train_data = np.memmap(
    args.train_bin,
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    args.val_bin,
    dtype=np.uint16,
    mode="r"
)

if len(train_data) <= args.seq + 1:
    raise ValueError("FineWeb training dataset is too small.")

if len(val_data) <= args.seq + 1:
    raise ValueError("FineWeb validation dataset is too small.")


# ------------------------------------------------------------
# 7A. REASONING DATA
# ------------------------------------------------------------

reasoning_data = None

n_reasoning_train = 0
reasoning_val_start = 0
reasoning_val_length = 0

if args.reasoning_pct > 0:

    if not os.path.exists(args.reasoning_bin):
        raise FileNotFoundError(args.reasoning_bin)

    reasoning_data = np.memmap(
        args.reasoning_bin,
        dtype=np.uint16,
        mode="r"
    )

    if len(reasoning_data) <= args.seq + args.reasoning_val_tokens:
        raise ValueError(
            "Reasoning dataset is too small for the requested "
            "training and validation split."
        )

    # Approximate target position for validation.
    target_start = max(
        0,
        len(reasoning_data) - args.reasoning_val_tokens
    )

    # Find the first EOS at or after target_start.
    # Validation begins immediately after that EOS.
    #
    # This prevents a training example from being split
    # between the training and validation regions.

    eos_positions = np.flatnonzero(
        reasoning_data[target_start:] == args.reasoning_eos_id
    )

    if len(eos_positions) == 0:
        raise RuntimeError(
            "Could not find an EOS token after the reasoning "
            "validation split position. Check reasoning_eos_id "
            "and your tokenized dataset."
        )

    reasoning_val_start = (
        target_start + int(eos_positions[0]) + 1
    )

    n_reasoning_train = reasoning_val_start

    reasoning_val_length = (
        len(reasoning_data) - reasoning_val_start
    )

    if n_reasoning_train <= args.seq + 1:
        raise ValueError("Reasoning training portion is too small.")

    if reasoning_val_length <= args.seq + 1:
        raise ValueError("Reasoning validation portion is too small.")

    write_log(
        f"Reasoning dataset tokens: {len(reasoning_data):,}"
    )

    write_log(
        f"Reasoning training tokens: {n_reasoning_train:,}"
    )

    write_log(
        f"Reasoning validation tokens: {reasoning_val_length:,}"
    )

    write_log(
        f"Reasoning validation starts at token: {reasoning_val_start:,}"
    )

    write_log(
        f"Reasoning EOS ID: {args.reasoning_eos_id}"
    )


# ============================================================
# 8. DATASET SAMPLING
# ============================================================

def is_reasoning_seq(i):

    # Exactly reasoning_pct percent over every 100 sequences.
    #
    # Example for 10%:
    # One reasoning sequence approximately every 10 sequences.
    #
    # The distribution is deterministic and resumes from seq_idx.

    return (
        ((i + 1) * args.reasoning_pct) // 100
        >
        (i * args.reasoning_pct) // 100
    )


# ------------------------------------------------------------
# 8A. UNSEEN FINEWEB BLOCKS
# ------------------------------------------------------------

order = None
unseen_blocks = None

sample_pos = 0
seq_idx = 0

needed = 0

if args.unseen.lower() != "none":

    if not os.path.exists(args.unseen):
        raise FileNotFoundError(
            f"{args.unseen} not found. "
            "Run scripts/make_block.py first."
        )

    unseen_blocks = np.load(args.unseen)

    if unseen_blocks.ndim != 1:
        raise ValueError("Unseen block file must be one-dimensional.")

    if len(unseen_blocks) == 0:
        raise ValueError("No unseen FineWeb blocks were found.")

    if not np.issubdtype(unseen_blocks.dtype, np.integer):
        raise ValueError("Unseen block indices must be integers.")

    # Number of blocks that can hold a full input sequence
    # plus its final target token.

    n_usable = (len(train_data) - 1) // args.seq

    if np.any(unseen_blocks < 0) or np.any(unseen_blocks >= n_usable):
        raise ValueError(
            "Unseen block file contains invalid block indices "
            "for the current FineWeb dataset or sequence length."
        )

    if len(np.unique(unseen_blocks)) != len(unseen_blocks):
        raise ValueError("Unseen block file contains duplicate indices.")

    order_rng = np.random.default_rng(args.seed)

    # Shuffle fresh blocks once.
    order = order_rng.permutation(unseen_blocks)

    if args.fill_seen:

        seen_blocks = np.setdiff1d(
            np.arange(n_usable, dtype=np.int64),
            unseen_blocks
        )

        order = np.concatenate([
            order,
            order_rng.permutation(seen_blocks)
        ])

    total_seqs = args.steps * args.accum * args.micro_bs

    needed = (
        total_seqs
        -
        (total_seqs * args.reasoning_pct) // 100
    )

    if needed > len(order):
        raise ValueError(
            f"Not enough FineWeb blocks.\n"
            f"Required: {needed:,}\n"
            f"Available: {len(order):,}\n"
            "Reduce steps or enable --fill_seen."
        )

else:

    write_log(
        "WARNING: FineWeb sampling uses random windows. "
        "Some pretraining data may repeat."
    )


# ============================================================
# 9. BATCH GENERATION
# ============================================================

def make_batch(data, indices):

    x = np.stack([
        data[i:i + args.seq] for i in indices
    ]).astype(np.int64)

    y = np.stack([
        data[i + 1:i + 1 + args.seq] for i in indices
    ]).astype(np.int64)

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    return x, y


def get_train_batch():

    global sample_pos, seq_idx

    xs = []
    ys = []

    for _ in range(args.micro_bs):

        if is_reasoning_seq(seq_idx):

            # Sample a random reasoning window.
            # The upper bound keeps the window inside the
            # training portion of the reasoning dataset.

            start = int(
                rng.integers(
                    0,
                    n_reasoning_train - args.seq - 1
                )
            )

            source = reasoning_data

        elif order is not None:

            # Each selected FineWeb block is used once.
            start = int(order[sample_pos]) * args.seq

            sample_pos += 1

            source = train_data

        else:

            # Random FineWeb sampling if unseen blocks are disabled.
            start = int(
                rng.integers(
                    0,
                    len(train_data) - args.seq - 1
                )
            )

            source = train_data

        xs.append(source[start:start + args.seq])

        ys.append(
            source[start + 1:start + 1 + args.seq]
        )

        seq_idx += 1

    x = torch.from_numpy(
        np.stack(xs).astype(np.int64)
    ).to(device)

    y = torch.from_numpy(
        np.stack(ys).astype(np.int64)
    ).to(device)

    return x, y


def get_val_batch(batch_rng):

    indices = batch_rng.integers(
        0,
        len(val_data) - args.seq - 1,
        size=args.micro_bs
    )

    return make_batch(val_data, indices)


def get_reasoning_val_batch(batch_rng):

    indices = (
        reasoning_val_start
        +
        batch_rng.integers(
            0,
            reasoning_val_length - args.seq - 1,
            size=args.micro_bs
        )
    )

    return make_batch(reasoning_data, indices)


# ============================================================
# 10. LOSS
# ============================================================

def compute_loss(x, y):

    with torch.autocast(
        device_type="cuda",
        dtype=amp_dtype,
        enabled=amp_enabled
    ):

        logits = model(x)

    loss = F.cross_entropy(
        logits.float().reshape(-1, cfg.vocab_size),
        y.reshape(-1)
    )

    return loss


# ============================================================
# 11. OPTIMIZER
# ============================================================

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


# ============================================================
# 12. LEARNING RATE SCHEDULER
# ============================================================

warmup = max(
    1,
    min(args.warmup_steps, args.steps)
)

min_lr_ratio = 0.1


def get_lr(step):

    if step < warmup:
        return args.lr * (step + 1) / warmup

    progress = (
        (step - warmup)
        /
        max(1, args.steps - warmup)
    )

    progress = min(1.0, max(0.0, progress))

    return args.lr * (
        min_lr_ratio
        +
        (1 - min_lr_ratio)
        * 0.5
        * (1 + math.cos(math.pi * progress))
    )


# ============================================================
# 13. VALIDATION
# ============================================================

@torch.no_grad()
def evaluate_fineweb(n_batches=50):

    model.eval()

    total = 0.0

    fixed_rng = np.random.default_rng(1)

    for _ in range(n_batches):

        x, y = get_val_batch(fixed_rng)

        loss = compute_loss(x, y)

        total += loss.item()

    model.train()

    return total / n_batches


@torch.no_grad()
def evaluate_reasoning(n_batches=20):

    if reasoning_data is None:
        return None

    model.eval()

    total = 0.0

    fixed_rng = np.random.default_rng(2)

    for _ in range(n_batches):

        x, y = get_reasoning_val_batch(fixed_rng)

        loss = compute_loss(x, y)

        total += loss.item()

    model.train()

    return total / n_batches


# ============================================================
# 14. CHECKPOINT STATE
# ============================================================

start_step = 0

best_fineweb_loss = float("inf")
best_reasoning_loss = float("inf")

last_fineweb_loss = None
last_reasoning_loss = None


# ============================================================
# 15. CHECKPOINT SAVING
# ============================================================

def save_checkpoint(path, step):

    torch.cuda.synchronize()

    checkpoint = {

        "model": model.state_dict(),

        "optimizer": opt.state_dict(),

        "cfg": asdict(cfg),

        "step": step,

        "val_loss": last_fineweb_loss,
        "best_val_loss": best_fineweb_loss,

        "fineweb_val_loss": last_fineweb_loss,
        "best_fineweb_loss": best_fineweb_loss,

        "reasoning_val_loss": last_reasoning_loss,
        "best_reasoning_loss": best_reasoning_loss,

        "train_rng_state": rng.bit_generator.state,
        "val_rng_state": val_rng.bit_generator.state,

        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),

        "training_history": training_history,

        "run_config": run_config,

        "sample_pos": sample_pos,
        "seq_idx": seq_idx,
    }

    temp_path = path + ".tmp"

    torch.save(checkpoint, temp_path)

    os.replace(temp_path, path)

    write_log(f"Checkpoint saved: {path}")


# ============================================================
# 16. INITIALIZE OR RESUME
# ============================================================

if args.resume is not None:

    if not os.path.exists(args.resume):
        raise FileNotFoundError(args.resume)

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
        "torch_rng_state",
        "cuda_rng_state",
        "run_config",
        "sample_pos",
        "seq_idx",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in checkpoint
    ]

    if missing_keys:
        raise RuntimeError(
            "Checkpoint is missing required resume information: "
            f"{missing_keys}"
        )

    if checkpoint["cfg"] != asdict(cfg):
        raise RuntimeError(
            "Model configuration differs from the checkpoint."
        )

    # Require identical settings on resume.
    # This prevents changing the data mixture or scheduler mid-run.

    saved_config = checkpoint["run_config"]

    if saved_config != run_config:
        differences = []

        for key in run_config:
            if saved_config.get(key) != run_config[key]:
                differences.append(
                    f"{key}: saved={saved_config.get(key)!r}, "
                    f"current={run_config[key]!r}"
                )

        raise RuntimeError(
            "Training configuration differs from the checkpoint:\n"
            + "\n".join(differences)
            + "\nUse the original settings when resuming."
        )

    model.load_state_dict(checkpoint["model"])

    opt.load_state_dict(checkpoint["optimizer"])

    start_step = int(checkpoint["step"])

    best_fineweb_loss = checkpoint.get(
        "best_fineweb_loss",
        checkpoint.get("best_val_loss", float("inf"))
    )

    best_reasoning_loss = checkpoint.get(
        "best_reasoning_loss",
        float("inf")
    )

    last_fineweb_loss = checkpoint.get(
        "fineweb_val_loss",
        checkpoint.get("val_loss")
    )

    last_reasoning_loss = checkpoint.get(
        "reasoning_val_loss"
    )

    rng.bit_generator.state = checkpoint["train_rng_state"]

    if "val_rng_state" in checkpoint:
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

    sample_pos = int(checkpoint["sample_pos"])
    seq_idx = int(checkpoint["seq_idx"])

    if start_step >= args.steps:
        raise ValueError(
            "Checkpoint has already reached the requested "
            "training step count."
        )

    expected_seq_idx = (
        start_step * args.accum * args.micro_bs
    )

    if seq_idx != expected_seq_idx:
        raise RuntimeError(
            "Checkpoint sequence counter does not match its step."
        )

    expected_sample_pos = (
        seq_idx
        -
        (seq_idx * args.reasoning_pct) // 100
    )

    if sample_pos != expected_sample_pos and order is not None:
        raise RuntimeError(
            "Checkpoint FineWeb sampling position is inconsistent."
        )

    write_log(f"Resuming from step {start_step:,}")


else:

    write_log(
        f"Training from scratch (random initialization), "
        f"{n_params:,} parameters, d_model={cfg.d_model}, "
        f"max_seq={cfg.max_seq}"
    )


# ============================================================
# 17. TRAINING INFORMATION
# ============================================================

print("\n========== FROM-SCRATCH TRAINING CONFIGURATION ==========")

print(f"Tag: {args.tag}")

print(
    "Initialization: "
    + ("random weights" if args.resume is None else f"resumed from {args.resume}")
)

print(f"Model parameters: {n_params:,}")

print(f"Total training steps: {args.steps:,}")
print(f"Starting step: {start_step:,}")

print(f"Micro batch: {args.micro_bs}")
print(f"Gradient accumulation: {args.accum}")
print(f"Sequence length: {args.seq}")

print(f"Tokens per step: {tokens_per_step:,}")
print(f"Planned total tokens: {total_tokens:,}")

print(f"FineWeb training data: {args.train_bin}")
print(f"FineWeb validation data: {args.val_bin}")

if order is not None:

    print(f"Unseen blocks: {len(unseen_blocks):,}")
    print(f"Blocks available: {len(order):,}")
    print(f"FineWeb blocks required: {needed:,}")

    print(
        f"FineWeb blocks remaining after planned run: "
        f"{len(order) - needed:,}"
    )

else:

    print("FineWeb sampling: random windows")

print(f"Reasoning percentage: {args.reasoning_pct}%")

if reasoning_data is not None:

    print(f"Reasoning data: {args.reasoning_bin}")
    print(f"Reasoning training tokens: {n_reasoning_train:,}")
    print(f"Reasoning validation tokens: {reasoning_val_length:,}")

print(f"Peak learning rate: {args.lr}")
print(f"Warmup steps: {warmup}")

print(f"AMP precision: {amp_name}")

print(f"Validation interval: {args.eval_every}")
print(f"Checkpoint interval: {args.save_every}")

print("======================================================\n")


# ============================================================
# 18. INITIAL BASELINE VALIDATION
# ============================================================

if start_step == 0:

    initial_fineweb = evaluate_fineweb()

    write_log(
        f">>> step 0 | FineWeb validation loss "
        f"{initial_fineweb:.4f} (starting weights)"
    )

    training_history["initial_fineweb_val_loss"] = initial_fineweb

    if reasoning_data is not None:

        initial_reasoning = evaluate_reasoning()

        write_log(
            f">>> step 0 | Reasoning validation loss "
            f"{initial_reasoning:.4f} (starting weights)"
        )

        training_history["initial_reasoning_val_loss"] = (
            initial_reasoning
        )

    save_metrics()


# ============================================================
# 19. TRAINING LOOP
# ============================================================

model.train()

torch.cuda.reset_peak_memory_stats()

t0 = time.time()

for step in range(start_step, args.steps):

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    lr = get_lr(step)

    for group in opt.param_groups:
        group["lr"] = lr

    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    opt.zero_grad(set_to_none=True)

    step_loss = 0.0

    for _ in range(args.accum):

        x, y = get_train_batch()

        loss = compute_loss(x, y) / args.accum

        loss.backward()

        step_loss += loss.detach()

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    # --------------------------------------------------------
    # Optimizer update
    # --------------------------------------------------------

    opt.step()

    global_step = step + 1

    # --------------------------------------------------------
    # Progress logging
    # --------------------------------------------------------

    should_log = (
        global_step % 10 == 0
        or global_step == args.steps
    )

    if should_log:

        elapsed = time.time() - t0

        completed_steps = global_step - start_step

        speed = (
            tokens_per_step * completed_steps
            /
            max(elapsed, 1e-8)
        )

        mem = (
            torch.cuda.max_memory_allocated()
            /
            1e9
        )

        eta_h = (
            elapsed
            /
            max(completed_steps, 1)
            *
            (args.steps - global_step)
            /
            3600
        )

        train_loss = step_loss.item()

        grad_value = grad_norm.item()

        message = (
            f"step {global_step:6d} | "
            f"loss {train_loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad {grad_value:.2f} | "
            f"{speed:,.0f} tok/s | "
            f"{mem:.2f} GB | "
            f"eta {eta_h:.1f} h"
        )

        write_log(message)

        training_history["metrics"].append({

            "step": global_step,

            "train_loss": train_loss,

            "learning_rate": lr,

            "grad_norm": grad_value,

            "tokens_per_second": speed,

            "gpu_memory_gb": mem,

            "fineweb_val_loss": None,

            "reasoning_val_loss": None,
        })

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    should_evaluate = (
        global_step % args.eval_every == 0
        or global_step == args.steps
    )

    if should_evaluate:

        last_fineweb_loss = evaluate_fineweb()

        write_log(
            f">>> step {global_step} | "
            f"FineWeb validation loss "
            f"{last_fineweb_loss:.4f}"
        )

        last_reasoning_loss = None

        if reasoning_data is not None:

            last_reasoning_loss = evaluate_reasoning()

            write_log(
                f">>> step {global_step} | "
                f"Reasoning validation loss "
                f"{last_reasoning_loss:.4f}"
            )

        # Update current metric record.

        if (
            training_history["metrics"]
            and training_history["metrics"][-1]["step"] == global_step
        ):

            training_history["metrics"][-1]["fineweb_val_loss"] = (
                last_fineweb_loss
            )

            training_history["metrics"][-1]["reasoning_val_loss"] = (
                last_reasoning_loss
            )

        # ----------------------------------------------------
        # BEST FINEWEB CHECKPOINT
        # ----------------------------------------------------

        if last_fineweb_loss < best_fineweb_loss:

            best_fineweb_loss = last_fineweb_loss

            save_checkpoint(
                best_fineweb_path,
                global_step
            )

            write_log(
                f"New best FineWeb validation loss: "
                f"{best_fineweb_loss:.4f}"
            )

        # ----------------------------------------------------
        # BEST REASONING CHECKPOINT
        # ----------------------------------------------------

        if (
            last_reasoning_loss is not None
            and last_reasoning_loss < best_reasoning_loss
        ):

            best_reasoning_loss = last_reasoning_loss

            save_checkpoint(
                best_reasoning_path,
                global_step
            )

            write_log(
                f"New best reasoning validation loss: "
                f"{best_reasoning_loss:.4f}"
            )

    # --------------------------------------------------------
    # LATEST CHECKPOINT
    # --------------------------------------------------------

    should_save = (
        global_step % args.save_every == 0
        or global_step == args.steps
    )

    if should_save:

        save_checkpoint(
            latest_path,
            global_step
        )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    if should_evaluate or should_save:
        save_metrics()


# ============================================================
# 20. TRAINING COMPLETE
# ============================================================

training_history["end_time"] = datetime.now().isoformat()

save_metrics()

write_log("\nTraining completed successfully.")

write_log(f"Latest checkpoint: {latest_path}")

write_log(f"Best FineWeb checkpoint: {best_fineweb_path}")

write_log(f"Best reasoning checkpoint: {best_reasoning_path}")

write_log(f"Training log: {log_file}")

write_log(f"Training metrics: {log_json}")

write_log("\nFinal validation results:")

write_log(
    f"Best FineWeb validation loss: {best_fineweb_loss:.4f}"
)

write_log(
    f"Best reasoning validation loss: {best_reasoning_loss:.4f}"
)