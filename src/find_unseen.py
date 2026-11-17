import argparse
import os

import numpy as np

# Replays the sampling of the ORIGINAL pretraining run (train.py, seed 0) to find
# which parts of train.bin the model never saw. The settings below must match
# that run: micro_bs 4, accum 16, seq 1024, 85,000 steps, seed 0.

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/tokenized/train.bin")
parser.add_argument("--out", default="data/tokenized/unseen_blocks.npy")
parser.add_argument("--steps", type=int, default=85000)
parser.add_argument("--accum", type=int, default=16)
parser.add_argument("--micro_bs", type=int, default=4)
parser.add_argument("--seq", type=int, default=1024)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

data = np.memmap(args.data, dtype=np.uint16, mode="r")
n = len(data)

# train.py drew window starts from [0, n - seq - 1)
hi = n - args.seq - 1

n_blocks = n // args.seq
seen = np.zeros(n_blocks + 2, dtype=bool)

rng = np.random.default_rng(args.seed)

# One get_batch() call per micro-batch, in the same order as the original run.
# (Validation used a separate generator, so it did not affect this stream.)
for _ in range(args.steps * args.accum):
    ix = rng.integers(0, hi, size=args.micro_bs)

    # A window covers tokens ix .. ix + seq (the +1 is the target of the last
    # input token), which always touches exactly blocks b and b + 1.
    b = ix // args.seq
    seen[b] = True
    seen[b + 1] = True

# A block is usable if the model never touched it and a full window fits in the file.
usable = np.flatnonzero(~seen[:n_blocks])
usable = usable[(usable + 1) * args.seq + 1 <= n]

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
np.save(args.out, usable.astype(np.int64))

print(f"tokens in file:        {n:,}")
print(f"blocks of {args.seq} tokens: {n_blocks:,}")
print(f"blocks the model saw:  {int(seen[:n_blocks].sum()):,} ({seen[:n_blocks].mean():.1%})")
print(f"unseen blocks:         {len(usable):,}  (~{len(usable) * args.seq / 1e9:.2f}B tokens)")
print(f"saved {args.out}")