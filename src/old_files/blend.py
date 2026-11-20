import argparse
import os

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--base", default="checkpoints/base_final.pt")
parser.add_argument("--sft", default="checkpoints/base_sft_best.pt")
parser.add_argument(
    "--weights",
    type=float,
    nargs="+",
    default=[0.3, 0.5, 0.7],
    help="Share of the SFT model in each blend. 0.0 = base, 1.0 = SFT.",
)
parser.add_argument("--out_dir", default="checkpoints")
args = parser.parse_args()

# ---------- load both checkpoints on the CPU ----------
base = torch.load(args.base, map_location="cpu", weights_only=False)
sft = torch.load(args.sft, map_location="cpu", weights_only=False)

# The two models must have the same shape, or blending makes no sense.
if base["cfg"] != sft["cfg"]:
    raise SystemExit("The two checkpoints have different configs.")

base_sd = base["model"]
sft_sd = sft["model"]

if base_sd.keys() != sft_sd.keys():
    raise SystemExit("The two checkpoints have different parameter names.")

os.makedirs(args.out_dir, exist_ok=True)

# ---------- blend ----------
for w in args.weights:
    if not 0.0 <= w <= 1.0:
        raise SystemExit(f"Weight {w} must be between 0 and 1.")

    blended = {}

    for name, sft_t in sft_sd.items():
        base_t = base_sd[name]

        if sft_t.is_floating_point():
            # (1 - w) * base + w * sft, computed in float32, stored in the original dtype
            mix = (1.0 - w) * base_t.float() + w * sft_t.float()
            blended[name] = mix.to(sft_t.dtype)
        else:
            # integer buffers (if any) cannot be averaged; take the SFT copy
            blended[name] = sft_t.clone()

    path = os.path.join(args.out_dir, f"blend_{w:.2f}.pt")

    torch.save(
        {
            "model": blended,
            "cfg": sft["cfg"],
            # a string here becomes the label in accuracy_comparison.py
            "step": f"blend{w:.2f}",
            # generate_sft.py prints val_loss with a number format, so keep it numeric
            "val_loss": float("nan"),
        },
        path,
    )

    print(f"saved {path}  ({(1 - w):.2f} base + {w:.2f} SFT)")