import torch
import torch.nn.functional as F
from model import Config, SLM

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
model = SLM(cfg).to(device)

data = torch.randint(5, cfg.vocab_size, (4,129), device=device)
x = data[:, :-1]
y = data[:, 1:]

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.0)
warmup_steps = 20

for step in range(301):
    lr = 1e-3 * min(1.0, (step + 1) / warmup_steps)
    for group in opt.param_groups:
        group["lr"] = lr

    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = model(x)
    loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.reshape(-1))


    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step % 20 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f} | grad norm {grad_norm:.2f} | lr {lr:.2e}")