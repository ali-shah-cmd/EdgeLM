import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 16384 # number of tokens tokenizer know
    d_model: int = 512 # size of token representation
    n_layers: int = 20 # number of transformer blocks
    n_heads: int = 8 # Query attention heads
    n_kv_heads: int = 2 # Key/Value attention heads
    head_dim: int = 64 # Attention head dimension d_model = n_heads * head_dim => 8 * 64 = 512
    ffn_dim: int = 1408 # Fead Forward Network dimension
    ple_dim: int = 32 # Size of PLE representation
    max_seq: int = 2048 # max supported sequence length
    rope_theta: float = 10000.0 # RoPE frequency
    norm_eps: float = 1e-6 # numerical stability constant (prevent division by zero)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean_square = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return x * self.weight

def build_rope(max_seq, head_dim, theta):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    angles = torch.outer(positions, inv_freq)
    angles = torch.cat([angles, angles], dim=1)
    return angles.cos(), angles.sin()

def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        c = self.cfg

        q = self.q_proj(x).view(B, T, c.n_heads, c.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, T, c.n_kv_heads, c.head_dim).transpose(1,2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        group = c.n_heads // c.n_kv_heads
        k = k.repeat_interleave(group, dim=1)
        v = v.repeat_interleave(group, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1,2).reshape(B,T, c.n_heads * c.head_dim)
        return self.o_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)
        self.ple_gate = nn.Linear(cfg.d_model, cfg.ple_dim, bias=False)
        self.ple_up = nn.Linear(cfg.ple_dim, cfg.d_model, bias=False)
        self.ple_norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, x, cos, sin, ple):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        g = F.gelu(self.ple_gate(x), approximate="tanh") * ple
        x = x + self.ple_norm(self.ple_up(g))
        return x

class SLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ple_table = nn.Embedding(cfg.vocab_size, cfg.n_layers * cfg.ple_dim)
        self.ple_proj = nn.Linear(cfg.d_model, cfg.n_layers * cfg.ple_dim, bias=False)
        self.ple_proj_norm = RMSNorm(cfg.ple_dim, cfg.norm_eps)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head= nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope(cfg.max_seq, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "down_proj.weight", "ple_up.weight")):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        B, T = ids.shape
        cfg = self.cfg

        x = self.tok_emb(ids)

        ple_tok = self.ple_table(ids).view(B, T, cfg.n_layers, cfg.ple_dim)
        ple_tok = ple_tok * math.sqrt(cfg.ple_dim)
        ple_ctx = self.ple_proj(x).view(B, T, cfg.n_layers, cfg.ple_dim)
        ple_ctx = self.ple_proj_norm(ple_ctx)
        ple = (ple_tok + ple_ctx) / math.sqrt(2.0)

        cos, sin = self.cos[:T], self.sin[:T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, ple[:, :, i])

        x = self.final_norm(x)
        return self.lm_head(x)


if __name__ == "__main__":
    cfg = Config()
    model = SLM(cfg)

    total = sum(p.numel() for p in model.parameters())
    ple_params = model.ple_table.weight.numel()
    print(f"total parameters : {total:,}")
    print(f"PLE lookup table : {ple_params:,}")
    print(f"without PLE table: {total - ple_params:,}")
    print("weights tied     :", model.lm_head.weight is model.tok_emb.weight)

    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    logits = model(ids)
    print("logits shape     :", tuple(logits.shape))

    targets = torch.randint(0, cfg.vocab_size, (2, 32))
    loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1))
    print(f"initial loss     : {loss.item():.3f}  (random guessing = {math.log(cfg.vocab_size):.3f})")