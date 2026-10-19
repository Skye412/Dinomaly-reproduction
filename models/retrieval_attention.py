"""Trainable linear cross-attention for retrieval-enhanced Dinomaly."""
import torch
from torch import nn
from torch.nn import functional as F


class RetrievalCrossAttention(nn.Module):
    """Linear cross-attention from bottleneck tokens to retrieved normal patches."""
    def __init__(self, dim=768, context_dim=768, heads=12, dropout=0.0):
        super().__init__()
        if dim % heads:
            raise ValueError('dim must be divisible by heads')
        self.heads, self.head_dim = heads, dim // heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(context_dim, dim, bias=True)
        self.v = nn.Linear(context_dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context):
        if context.ndim != 3 or context.shape[0] != x.shape[0] or context.shape[-1] != x.shape[-1]:
            raise ValueError(f'Invalid retrieval context shape {tuple(context.shape)} for {tuple(x.shape)}')
        b, nq, d = x.shape
        q = self.q(self.norm_q(x)).view(b, nq, self.heads, self.head_dim).transpose(1, 2)
        normalized_context = self.norm_context(context)
        k = self.k(normalized_context).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(normalized_context).view(b, -1, self.heads, self.head_dim).transpose(1, 2)
        # Kernelized linear attention: phi(z)=elu(z)+1, preserving decoder's O(N) scaling.
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        denom = 1.0 / (torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2)) + 1e-6)
        y = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, denom)
        y = y.transpose(1, 2).reshape(b, nq, d)
        result = self.norm_out(x + self.dropout(self.proj(y)))
        if not torch.isfinite(result).all():
            raise RuntimeError('Non-finite retrieval cross-attention output')
        return result
