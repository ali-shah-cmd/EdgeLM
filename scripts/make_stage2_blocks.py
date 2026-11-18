import numpy as np
import torch

ckpt = torch.load("checkpoints/reasoningcont_latest.pt", map_location="cpu", weights_only=False)
cfg = ckpt["run_config"]
assert not cfg["fill_seen"], "Main run used --fill_seen; this script does not handle that."

blocks = np.load(cfg["unseen"])
order = np.random.default_rng(cfg["seed"]).permutation(blocks)

total_seqs = cfg["steps"] * cfg["accum"] * cfg["micro_bs"]
needed = total_seqs - (total_seqs * cfg["reasoning_pct"]) // 100

rest = order[needed:]
np.save("data/tokenized/unseen_blocks_stage2.npy", rest)
print(f"main run uses {needed:,} blocks; leftover saved: {len(rest):,}")