"""Inspect a training checkpoint and save a weights-only copy.

Usage (from C:\\Work\\SLM, venv active):
    python scripts\\slim_checkpoint.py --ckpt checkpoints\\reasoningcont_latest.pt

Output: checkpoints\\reasoningcont_latest_weights.pt (unless --out is given)
"""
import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def tensor_bytes(obj):
    """Total size in bytes of all tensors inside a (nested) object."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt_path = ROOT / args.ckpt
    out_path = ROOT / args.out if args.out else ckpt_path.with_name(ckpt_path.stem + "_weights.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {out_path}")

    print(f"Loading {ckpt_path} ...")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("\nWhat is inside the file (tensor data only):")
    for key, value in ck.items():
        print(f"  {key:14s} {tensor_bytes(value) / 1e6:9.1f} MB")

    keep = [k for k in ("model", "cfg", "step") if k in ck]
    slim = {k: ck[k] for k in keep}
    print(f"\nKeeping keys: {keep}")
    torch.save(slim, out_path)
    print(f"Saved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB on disk)")
    print(f"Original: {ckpt_path.stat().st_size / 1e6:.1f} MB on disk")

    # Verify: reload the slim file and compare every weight tensor exactly.
    print("\nVerifying saved weights are identical ...")
    check = torch.load(out_path, map_location="cpu", weights_only=False)
    same = (
        check["model"].keys() == ck["model"].keys()
        and all(torch.equal(check["model"][k], ck["model"][k]) for k in ck["model"])
    )
    print("Weights identical:", same)


if __name__ == "__main__":
    main()