
import torch
from pathlib import Path

checkpoint_path = Path("checkpoints/ckpt_step_500.pt")

print("Loading checkpoint:", checkpoint_path)

# Only load checkpoints that you trust.
ckpt = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:", type(ckpt))

if isinstance(ckpt, dict):
    print("\nTop-level keys:")
    for key in ckpt:
        print(" -", key)

    # Find the model state dictionary
    state = None

    for key in ["model", "model_state_dict", "state_dict"]:
        if key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            print("\nModel weights found under:", key)
            break

    if state is None:
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
            print("\nCheckpoint itself appears to be a state_dict.")

    if state is not None:
        total_params = 0
        dtype_counts = {}

        print("\nMODEL TENSORS")
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                continue

            total_params += tensor.numel()
            dtype_counts[str(tensor.dtype)] = (
                dtype_counts.get(str(tensor.dtype), 0)
                + tensor.numel()
            )

            print(
                f"{name:50} "
                f"shape={str(tuple(tensor.shape)):25} "
                f"dtype={tensor.dtype}"
            )

        print("\nTotal parameters:", f"{total_params:,}")
        print("\nParameter counts by dtype:")
        for dtype, count in dtype_counts.items():
            print(f"{dtype}: {count:,}")

    print("\nTRAINING METADATA")
    for key in ["step", "epoch", "best_val_loss", "val_loss", "config"]:
        if key in ckpt:
            print(f"{key}: {ckpt[key]}")

else:
    print("Unexpected checkpoint format.")
