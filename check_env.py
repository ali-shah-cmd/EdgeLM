import torch

print("torch        :", torch.__version__)
print("cuda build   :", torch.version.cuda)
print("gpu available:", torch.cuda.is_available())

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("gpu          :", p.name)
    print("vram         :", round(p.total_memory / 1e9, 1), "GB")
    print("bf16 support :", torch.cuda.is_bf16_supported())
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("matmul ok    :", (a @ a).float().abs().mean().item() > 0)
else:
    print("-> torch can't see the GPU; re-check the driver and the cu128 install")


