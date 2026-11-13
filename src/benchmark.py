
import time
import torch

from model import Config, SLM

CHECKPOINT = "checkpoints/ckpt.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------
# Load model
# --------------------------------

start = time.perf_counter()

ckpt = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

cfg = Config(**ckpt["cfg"])

model = SLM(cfg).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

synchronize()
load_time = time.perf_counter() - start

print("\n========== MODEL INFO ==========")
print(f"Checkpoint step : {ckpt['step']}")
print(f"Validation loss : {ckpt['val_loss']:.4f}")
print(f"Device          : {device}")
print(f"Parameters      : {sum(p.numel() for p in model.parameters()):,}")
print(f"Load time       : {load_time:.3f} seconds")


# --------------------------------
# Prefill benchmark
# --------------------------------

def benchmark_prefill(seq_len, runs=10):

    x = torch.randint(
        0,
        cfg.vocab_size,
        (1, seq_len),
        device=device
    )

    # Warmup
    with torch.inference_mode():
        for _ in range(3):
            model(x)

    synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(runs):
            model(x)

    synchronize()

    elapsed = time.perf_counter() - start

    avg_latency = elapsed / runs
    tokens_per_sec = seq_len / avg_latency

    print(
        f"Prompt: {seq_len:4d} tokens | "
        f"Latency: {avg_latency * 1000:8.2f} ms | "
        f"Throughput: {tokens_per_sec:10.2f} tokens/sec"
    )


print("\n========== PREFILL BENCHMARK ==========")

sequence_lengths = [
    32, 128, 512, 1024
]

for seq_len in sequence_lengths:
    if seq_len <= cfg.max_seq:
        benchmark_prefill(seq_len)


# --------------------------------
# Generation benchmark
# --------------------------------

def generate_tokens(x, n_tokens):

    with torch.inference_mode():

        for _ in range(n_tokens):

            logits = model(
                x[:, -cfg.max_seq:]
            )[:, -1, :]

            next_id = torch.argmax(
                logits,
                dim=-1,
                keepdim=True
            )

            x = torch.cat(
                [x, next_id],
                dim=1
            )

    return x


print("\n========== GENERATION BENCHMARK ==========")

prompt = torch.randint(
    0,
    cfg.vocab_size,
    (1, 32),
    device=device
)

# Warmup
generate_tokens(prompt.clone(), 5)

synchronize()

n_tokens = 50

start = time.perf_counter()

output = generate_tokens(
    prompt.clone(),
    n_tokens
)

synchronize()

elapsed = time.perf_counter() - start

print(f"Generated tokens : {n_tokens}")
print(f"Generation time  : {elapsed:.3f} seconds")
print(f"Average latency  : {elapsed / n_tokens * 1000:.2f} ms/token")
print(f"Generation speed : {n_tokens / elapsed:.2f} tokens/sec")


# --------------------------------
# GPU memory benchmark
# --------------------------------

if device.type == "cuda":

    allocated = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2

    print("\n========== GPU MEMORY ==========")
    print(f"Peak allocated : {allocated:.2f} MB")
    print(f"Peak reserved  : {reserved:.2f} MB")


print("\nBenchmark completed.")
