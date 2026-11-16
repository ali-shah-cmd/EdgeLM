import os
import gc
import time
import threading
import psutil
import torch

from model import Config, SLM


CHECKPOINT = "checkpoints/ckpt.pt"

PREFILL_LENGTHS = [32, 128, 512, 1024]
GENERATION_TOKENS = 50
BATCH_SIZE = 1

process = psutil.Process(os.getpid())


# --------------------------------------------------
# 1. CPU RAM MONITOR
# --------------------------------------------------

class MemoryMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak = 0
        self.running = False
        self.thread = None

    def sample(self):
        while self.running:
            try:
                memory = process.memory_info().rss
                self.peak = max(self.peak, memory)
            except psutil.Error:
                pass
            time.sleep(self.interval)

    def start(self):
        self.peak = process.memory_info().rss
        self.running = True
        self.thread = threading.Thread(
            target=self.sample,
            daemon=True
        )
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()


def mb(value):
    return value / (1024 ** 2)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------
# 2. MODEL CONFIGURATION / KV CACHE ESTIMATE
# --------------------------------------------------

def get_config_value(cfg, names):
    for name in names:
        if hasattr(cfg, name):
            value = getattr(cfg, name)
            if value is not None:
                return value
    return None


def estimate_kv_cache(cfg, dtype, batch_size, sequence_length):
    """
    Estimates a standard Transformer KV cache.

    Formula:
    2 * layers * batch * sequence_length
      * KV_heads * head_dim * bytes_per_element

    This is theoretical. Your current generation code
    does not actually allocate a KV cache.
    """

    layers = get_config_value(
        cfg, ["n_layer", "num_layers", "n_layers"]
    )

    heads = get_config_value(
        cfg, ["n_head", "num_heads", "n_heads"]
    )

    kv_heads = get_config_value(
        cfg, ["n_kv_head", "num_kv_heads", "n_kv_heads"]
    )

    embedding_dim = get_config_value(
        cfg, ["n_embd", "d_model", "hidden_size", "dim"]
    )

    head_dim = get_config_value(
        cfg, ["head_dim", "d_head"]
    )

    if kv_heads is None:
        kv_heads = heads

    if head_dim is None and embedding_dim and heads:
        head_dim = embedding_dim // heads

    if not all([layers, kv_heads, head_dim]):
        return None

    bytes_per_element = torch.empty(
        (), dtype=dtype
    ).element_size()

    cache_bytes = (
        2
        * layers
        * batch_size
        * sequence_length
        * kv_heads
        * head_dim
        * bytes_per_element
    )

    return cache_bytes


# --------------------------------------------------
# 3. PREFILL BENCHMARK
# --------------------------------------------------

def benchmark_prefill(model, cfg, device, seq_len, baseline_allocated):
    x = torch.randint(
        0,
        cfg.vocab_size,
        (BATCH_SIZE, seq_len),
        device=device
    )

    runs = 10 if device.type == "cuda" else 5

    with torch.inference_mode():
        for _ in range(3):
            model(x)

    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(runs):
            model(x)

    synchronize(device)

    elapsed = time.perf_counter() - start
    avg_latency = elapsed / runs
    tokens_per_sec = (BATCH_SIZE * seq_len) / avg_latency

    print(f"\nPrefill: {seq_len} tokens")
    print(f"  Average latency       : {avg_latency * 1000:.2f} ms")
    print(f"  Throughput            : {tokens_per_sec:.2f} tokens/sec")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        extra_allocated = max(
            0, peak_allocated - baseline_allocated
        )

        allocator_overhead = max(
            0, peak_reserved - peak_allocated
        )

        print(f"  Peak GPU allocated    : {mb(peak_allocated):.2f} MB")
        print(f"  Extra forward memory  : {mb(extra_allocated):.2f} MB")
        print(f"  Peak GPU reserved     : {mb(peak_reserved):.2f} MB")
        print(f"  Reserved - allocated  : {mb(allocator_overhead):.2f} MB")

    return avg_latency


# --------------------------------------------------
# 4. GENERATION BENCHMARK
# --------------------------------------------------

def generate_tokens(model, cfg, prompt, n_tokens):
    x = prompt.clone()

    with torch.inference_mode():
        for _ in range(n_tokens):
            # No KV cache: recomputes the full context.
            logits = model(x[:, -cfg.max_seq:])[:, -1, :]

            next_id = torch.argmax(
                logits, dim=-1, keepdim=True
            )

            x = torch.cat([x, next_id], dim=1)

    return x


def benchmark_generation(model, cfg, device, baseline_allocated):
    prompt = torch.randint(
        0,
        cfg.vocab_size,
        (BATCH_SIZE, 32),
        device=device
    )

    # Warmup
    generate_tokens(model, cfg, prompt, 5)
    synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()

    output = generate_tokens(
        model,
        cfg,
        prompt,
        GENERATION_TOKENS
    )

    synchronize(device)

    elapsed = time.perf_counter() - start

    tokens_per_sec = GENERATION_TOKENS / elapsed
    latency_per_token = elapsed / GENERATION_TOKENS

    print("\nGeneration benchmark")
    print(f"  Generated tokens     : {GENERATION_TOKENS}")
    print(f"  Total time           : {elapsed:.3f} seconds")
    print(f"  Latency per token    : {latency_per_token * 1000:.2f} ms")
    print(f"  Generation speed     : {tokens_per_sec:.2f} tokens/sec")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        extra_allocated = max(
            0, peak_allocated - baseline_allocated
        )

        print(f"  Peak GPU allocated   : {mb(peak_allocated):.2f} MB")
        print(f"  Extra generation mem : {mb(extra_allocated):.2f} MB")
        print(f"  Peak GPU reserved    : {mb(peak_reserved):.2f} MB")

    del output


# --------------------------------------------------
# 5. COMPLETE DEVICE BENCHMARK
# --------------------------------------------------

def benchmark(device):
    print("\n" + "=" * 55)
    print(f"BENCHMARKING ON: {device}")
    print("=" * 55)

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ram_monitor = MemoryMonitor()
    ram_monitor.start()

    ram_before = process.memory_info().rss

    # Load checkpoint
    load_start = time.perf_counter()

    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    cfg = Config(**ckpt["cfg"])
    checkpoint_step = ckpt.get("step", "unknown")
    model = SLM(cfg)

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    synchronize(device)

    load_time = time.perf_counter() - load_start

    # Parameter and buffer memory
    parameter_count = sum(
        p.numel() for p in model.parameters()
    )

    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    parameter_dtypes = sorted({
        str(p.dtype) for p in model.parameters()
    })

    model_memory_bytes = parameter_bytes + buffer_bytes

    checkpoint_size = os.path.getsize(CHECKPOINT)

    # Release checkpoint dictionary
    del ckpt
    gc.collect()

    ram_after_load = process.memory_info().rss

    if device.type == "cuda":
        baseline_allocated = torch.cuda.memory_allocated(device)
        baseline_reserved = torch.cuda.memory_reserved(device)

        free_gpu, total_gpu = torch.cuda.mem_get_info(device)
    else:
        baseline_allocated = 0
        baseline_reserved = 0
        free_gpu = total_gpu = None

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    print("\nMODEL INFORMATION")
    print(f"  Checkpoint file       : {CHECKPOINT}")
    print(f"  Checkpoint size       : {mb(checkpoint_size):.2f} MB")
    # print(f"  Training step         : {getattr(model, 'step', 'see checkpoint')}")
    print(f"  Training step         : {checkpoint_step}")
    print(f"  Parameters            : {parameter_count:,}")
    print(f"  Parameter dtype(s)    : {parameter_dtypes}")
    print(f"  Parameter memory      : {mb(parameter_bytes):.2f} MB")
    print(f"  Model buffer memory   : {mb(buffer_bytes):.2f} MB")
    print(f"  Total model tensors   : {mb(model_memory_bytes):.2f} MB")
    print(f"  Model load time       : {load_time:.3f} seconds")

    # --------------------------------------------------
    # CPU RAM
    # --------------------------------------------------

    print("\nCPU RAM")
    print(f"  Before loading        : {mb(ram_before):.2f} MB")
    print(f"  After loading         : {mb(ram_after_load):.2f} MB")
    print(f"  Increase after load   : {mb(max(0, ram_after_load - ram_before)):.2f} MB")

    # --------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------

    if device.type == "cuda":
        print("\nGPU MEMORY")
        print(f"  Total VRAM            : {mb(total_gpu):.2f} MB")
        print(f"  Free VRAM at start    : {mb(free_gpu):.2f} MB")
        print(f"  Model allocated      : {mb(baseline_allocated):.2f} MB")
        print(f"  Model reserved       : {mb(baseline_reserved):.2f} MB")

    # --------------------------------------------------
    # KV CACHE ESTIMATES
    # --------------------------------------------------

    print("\nTHEORETICAL KV CACHE ESTIMATES")

    dtype = next(model.parameters()).dtype

    for seq_len in [128, 512, 1024]:
        cache_bytes = estimate_kv_cache(
            cfg,
            dtype,
            BATCH_SIZE,
            seq_len
        )

        if cache_bytes is None:
            print(
                f"  {seq_len} tokens: unavailable "
                "(could not identify layer/head dimensions)"
            )
        else:
            print(
                f"  {seq_len} tokens: "
                f"{mb(cache_bytes):.2f} MB"
            )

    print("  Note: these are estimates, not measured allocations.")
    print("  Current generation implementation does not use a KV cache.")

    # --------------------------------------------------
    # PREFILL
    # --------------------------------------------------

    print("\nPREFILL BENCHMARK")

    for seq_len in PREFILL_LENGTHS:
        if seq_len <= cfg.max_seq:
            benchmark_prefill(
                model,
                cfg,
                device,
                seq_len,
                baseline_allocated
            )

    # --------------------------------------------------
    # GENERATION
    # --------------------------------------------------

    benchmark_generation(
        model,
        cfg,
        device,
        baseline_allocated
    )

    # --------------------------------------------------
    # FINAL MEMORY
    # --------------------------------------------------

    ram_monitor.stop()

    ram_peak = max(
        ram_monitor.peak,
        process.memory_info().rss
    )

    print("\nFINAL MEMORY SUMMARY")
    print(f"  Current process RAM   : {mb(process.memory_info().rss):.2f} MB")
    print(f"  Peak process RAM      : {mb(ram_peak):.2f} MB")
    print(f"  Peak RAM increase     : {mb(max(0, ram_peak - ram_before)):.2f} MB")

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)

        print(f"  Peak GPU allocated    : {mb(peak_allocated):.2f} MB")
        print(f"  Peak GPU reserved     : {mb(peak_reserved):.2f} MB")
        print(
            f"  Peak extra GPU memory : "
            f"{mb(max(0, peak_allocated - baseline_allocated)):.2f} MB"
        )

    del model
    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()


# --------------------------------------------------
# 6. RUN CPU AND GPU BENCHMARKS
# --------------------------------------------------

benchmark(torch.device("cpu"))

if torch.cuda.is_available():
    benchmark(torch.device("cuda"))
else:
    print("\nCUDA is not available.")
