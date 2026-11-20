# """Print a full report of the model: config, every weight matrix, size per layer,
# activation sizes per module, and total size at different precisions.

# Usage (from C:\\Work\\SLM, venv active):
#     python scripts\\inspect_model.py --ckpt checkpoints\\reasoningcont_latest_weights.pt
#     python scripts\\inspect_model.py --ckpt checkpoints\\reasoningcont_latest.pt --seq 256 --out eval_results\\model_report.txt
# """
# import argparse
# import sys
# from collections import OrderedDict
# from pathlib import Path

# import torch

# ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT))

# from src.model import Config, SLM  # same imports as src\eval_reasoning.py

# LINES = []


# def out(text=""):
#     print(text)
#     LINES.append(text)


# def mb(n_bytes):
#     return n_bytes / 1e6


# def group_name(param_name):
#     """'blocks.3.attn.q_proj.weight' -> 'blocks.3'; 'tok_emb.weight' -> 'tok_emb'."""
#     parts = param_name.split(".")
#     for i, p in enumerate(parts):
#         if p.isdigit():
#             return ".".join(parts[: i + 1])
#     return parts[0]


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ckpt", required=True)
#     parser.add_argument("--seq", type=int, default=128,
#                         help="Sequence length used for the activation-size test.")
#     parser.add_argument("--out", default=None, help="Optional text file to save the report.")
#     args = parser.parse_args()

#     ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
#     cfg = ck["cfg"]
#     model = SLM(Config(**cfg))
#     model.load_state_dict(ck["model"])
#     model.eval()

#     # ---------------- 1. config ----------------
#     out("=" * 70)
#     out("CONFIG (from the checkpoint)")
#     out("=" * 70)
#     for k, v in cfg.items():
#         out(f"  {k:20s} {v}")
#     out(f"  {'training step':20s} {ck.get('step', 'unknown')}")

#     # ---------------- 2. every weight tensor ----------------
#     out("\n" + "=" * 70)
#     out("EVERY WEIGHT TENSOR  (name | shape | parameters | MB in fp32)")
#     out("=" * 70)
#     groups = OrderedDict()
#     total_params = 0
#     for name, p in model.named_parameters():
#         n = p.numel()
#         total_params += n
#         out(f"  {name:55s} {str(tuple(p.shape)):18s} {n:>12,d} {mb(n * 4):8.2f}")
#         groups.setdefault(group_name(name), [0, 0])
#         groups[group_name(name)][0] += n
#         groups[group_name(name)][1] += 1

#     # ---------------- 3. per-layer totals ----------------
#     out("\n" + "=" * 70)
#     out("SIZE PER LAYER / GROUP")
#     out("=" * 70)
#     out(f"  {'group':25s} {'tensors':>8s} {'parameters':>14s} {'% of model':>11s} {'fp32 MB':>9s}")
#     for g, (n, count) in groups.items():
#         out(f"  {g:25s} {count:8d} {n:14,d} {100 * n / total_params:10.2f}% {mb(n * 4):9.2f}")

#     # ---------------- 4. total at different precisions ----------------
#     out("\n" + "=" * 70)
#     out("TOTAL MODEL SIZE (weights only)")
#     out("=" * 70)
#     out(f"  parameters: {total_params:,d}")
#     for label, bits in (("fp32", 32), ("fp16 / bf16", 16), ("int8", 8), ("int4", 4)):
#         out(f"  {label:12s} {mb(total_params * bits / 8):8.1f} MB")
#     out("  (shared/tied weights are counted once; quantization scales are not included)")

#     # ---------------- 5. activation sizes ----------------
#     out("\n" + "=" * 70)
#     out(f"ACTIVATIONS: output of every leaf module for 1 x {args.seq} tokens (fp32)")
#     out("=" * 70)
#     records = []
#     hooks = []

#     def make_hook(name):
#         def hook(module, inputs, output):
#             if torch.is_tensor(output):
#                 records.append((name, type(module).__name__, tuple(output.shape),
#                                 output.numel() * output.element_size()))
#         return hook

#     for name, module in model.named_modules():
#         if name and len(list(module.children())) == 0:
#             hooks.append(module.register_forward_hook(make_hook(name)))

#     ids = torch.randint(5, 1000, (1, args.seq))  # ids 0-4 are special tokens
#     with torch.inference_mode():
#         logits = model(ids)
#     for h in hooks:
#         h.remove()

#     out(f"  {'module':45s} {'type':14s} {'output shape':22s} {'KB':>9s}")
#     for name, kind, shape, size in records:
#         out(f"  {name:45s} {kind:14s} {str(shape):22s} {size / 1e3:9.1f}")
#     biggest = max(records, key=lambda r: r[3])
#     out(f"\n  largest single output: {biggest[0]} {biggest[2]} = {biggest[3] / 1e6:.2f} MB")
#     out("  The sum of all outputs is NOT the peak memory: most are freed after use.")
#     out("  Activations scale linearly with sequence length (--seq).")
#     if torch.is_tensor(logits):
#         out(f"  final logits shape: {tuple(logits.shape)}")

#     if args.out:
#         path = ROOT / args.out
#         path.parent.mkdir(parents=True, exist_ok=True)
#         path.write_text("\n".join(LINES), encoding="utf-8")
#         print(f"\nReport saved to: {path}")


# if __name__ == "__main__":
#     main()


"""Print a full report of the model: config, every weight matrix, size per layer,
activation sizes per module, and total size at different precisions.

Usage (from C:\\Work\\SLM, venv active):
    # 1) from a trained checkpoint
    python scripts\\inspect_model.py --ckpt checkpoints\\reasoningcont_latest_weights.pt
    python scripts\\inspect_model.py --ckpt checkpoints\\reasoningcont_latest.pt --seq 256 --out eval_results\\model_report.txt

    # 2) no checkpoint: build a fresh random-weight model from the config arguments
    python scripts\\inspect_model.py --d_model 128 --seq 256 --out eval_results\\model_report_d128.txt
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import Config, SLM  # same imports as src\eval_reasoning.py

LINES = []


def out(text=""):
    print(text)
    LINES.append(text)


def mb(n_bytes):
    return n_bytes / 1e6


def group_name(param_name):
    """'blocks.3.attn.q_proj.weight' -> 'blocks.3'; 'tok_emb.weight' -> 'tok_emb'."""
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p.isdigit():
            return ".".join(parts[: i + 1])
    return parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Optional. If omitted, a fresh random-weight model is built from the args below.")
    # Config used only when --ckpt is not given (defaults = your old model, except the d_model-related ones)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--d_model", type=int, default=640)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_kv_heads", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--ffn_dim", type=int, default=1408)
    parser.add_argument("--ple_dim", type=int, default=32)
    parser.add_argument("--max_seq", type=int, default=16384)
    parser.add_argument("--seq", type=int, default=128,
                        help="Sequence length used for the activation-size test.")
    parser.add_argument("--out", default=None, help="Optional text file to save the report.")
    args = parser.parse_args()

    if args.ckpt:
        ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
        model = SLM(Config(**cfg))
        model.load_state_dict(ck["model"])
        step = ck.get("step", "unknown")
    else:
        cfg = dict(
            vocab_size=args.vocab_size, d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, n_kv_heads=args.n_kv_heads, head_dim=args.head_dim,
            ffn_dim=args.ffn_dim, ple_dim=args.ple_dim, max_seq=args.max_seq,
            rope_theta=10000.0, norm_eps=1e-6,
        )
        model = SLM(Config(**cfg))   # weights are random: fine, the report only needs shapes
        step = "none (random init, untrained)"
    model.eval()

    # ---------------- 1. config ----------------
    out("=" * 70)
    out("CONFIG (from the checkpoint)" if args.ckpt else "CONFIG (from command-line args, random weights)")
    out("=" * 70)
    for k, v in cfg.items():
        out(f"  {k:20s} {v}")
    out(f"  {'training step':20s} {step}")

    # ---------------- 2. every weight tensor ----------------
    out("\n" + "=" * 70)
    out("EVERY WEIGHT TENSOR  (name | shape | parameters | MB in fp32)")
    out("=" * 70)
    groups = OrderedDict()
    total_params = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total_params += n
        out(f"  {name:55s} {str(tuple(p.shape)):18s} {n:>12,d} {mb(n * 4):8.2f}")
        groups.setdefault(group_name(name), [0, 0])
        groups[group_name(name)][0] += n
        groups[group_name(name)][1] += 1

    # ---------------- 3. per-layer totals ----------------
    out("\n" + "=" * 70)
    out("SIZE PER LAYER / GROUP")
    out("=" * 70)
    out(f"  {'group':25s} {'tensors':>8s} {'parameters':>14s} {'% of model':>11s} {'fp32 MB':>9s}")
    for g, (n, count) in groups.items():
        out(f"  {g:25s} {count:8d} {n:14,d} {100 * n / total_params:10.2f}% {mb(n * 4):9.2f}")

    # ---------------- 4. total at different precisions ----------------
    out("\n" + "=" * 70)
    out("TOTAL MODEL SIZE (weights only)")
    out("=" * 70)
    out(f"  parameters: {total_params:,d}")
    for label, bits in (("fp32", 32), ("fp16 / bf16", 16), ("int8", 8), ("int4", 4)):
        out(f"  {label:12s} {mb(total_params * bits / 8):8.1f} MB")
    out("  (shared/tied weights are counted once; quantization scales are not included)")

    # ---------------- 5. activation sizes ----------------
    out("\n" + "=" * 70)
    out(f"ACTIVATIONS: output of every leaf module for 1 x {args.seq} tokens (fp32)")
    out("=" * 70)
    records = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, output):
            if torch.is_tensor(output):
                records.append((name, type(module).__name__, tuple(output.shape),
                                output.numel() * output.element_size()))
        return hook

    for name, module in model.named_modules():
        if name and len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(make_hook(name)))

    ids = torch.randint(5, 1000, (1, args.seq))  # ids 0-4 are special tokens
    with torch.inference_mode():
        logits = model(ids)
    for h in hooks:
        h.remove()

    out(f"  {'module':45s} {'type':14s} {'output shape':22s} {'KB':>9s}")
    for name, kind, shape, size in records:
        out(f"  {name:45s} {kind:14s} {str(shape):22s} {size / 1e3:9.1f}")
    biggest = max(records, key=lambda r: r[3])
    out(f"\n  largest single output: {biggest[0]} {biggest[2]} = {biggest[3] / 1e6:.2f} MB")
    out("  The sum of all outputs is NOT the peak memory: most are freed after use.")
    out("  Activations scale linearly with sequence length (--seq).")
    if torch.is_tensor(logits):
        out(f"  final logits shape: {tuple(logits.shape)}")

    if args.out:
        path = ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(LINES), encoding="utf-8")
        print(f"\nReport saved to: {path}")


if __name__ == "__main__":
    main()