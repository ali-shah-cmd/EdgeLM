import numpy as np

SEQ = 4096
data = np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode="r")

n_usable = (len(data) - 1) // SEQ
blocks = np.arange(n_usable, dtype=np.int64)

np.save("data/tokenized/all_blocks_4096.npy", blocks)
print(f"tokens: {len(data):,} | blocks of {SEQ}: {n_usable:,}")