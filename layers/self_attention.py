import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.1,
        mask: bool = True,
    ):
        """
        Multi-head self attention layer.

        d_k is the length of each key vector per head.
        If not provided, it is set to embed_dim // num_heads,
        in which case embed_dim must be devisible by num_heads.
        Likewise for d_v.

        If mask is False, this layer is an encoder block. If mask is True,
        this layer is a decoder block, preventing backwards dependencies
        by setting the attention weights above the diagonal to 0.
        """
        super(SelfAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if d_k is None or d_v is None:
            assert (
                embed_dim % num_heads == 0
            ), "Embedding dimension must be divisible by number of heads"

        self.d_k = d_k if d_k is not None else embed_dim // num_heads
        self.d_v = d_v if d_v is not None else embed_dim // num_heads

        # d_q must be equal to d_k to compute dot product between Q and K
        self.d_q = self.d_k

        self.q_proj_weight = nn.Linear(embed_dim, self.d_q * num_heads, bias=False)
        self.k_proj_weight = nn.Linear(embed_dim, self.d_k * num_heads, bias=False)
        self.v_proj_weight = nn.Linear(embed_dim, self.d_v * num_heads, bias=False)

        self.is_masked = mask
        if mask:
            mask = torch.tril(torch.ones((embed_dim, embed_dim)))
            self.register_buffer("mask", mask)

        self.fc_out = nn.Linear(self.d_v * num_heads, embed_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N = batch size
        # L = sequence length
        N, L, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Input tensor has embedding dimension {embed_dim}, "
                f"expected {self.embed_dim}"
            )

        # Project Q, K, and V per head in parallel. The heads are physically
        # contiguous in memory, but logically separate. We reshape to logical
        # groupings after the projection.
        q: torch.Tensor = self.q_proj_weight(x)
        k: torch.Tensor = self.k_proj_weight(x)
        v: torch.Tensor = self.v_proj_weight(x)

        # Reshape to logical groupings, yielding Q, K, and V for each head.
        q = q.view(N, self.num_heads, L, self.d_q)
        k = k.view(N, self.num_heads, L, self.d_k)
        v = v.view(N, self.num_heads, L, self.d_v)

        # Batch matrix multiply Q and K per head, scaling by sqrt(d_k)
        # to prevent softmax from saturating later on.
        qk_t: torch.Tensor = q @ k.transpose(-2, -1) / (self.d_k**0.5)

        qk_t = self.dropout(qk_t)

        if self.is_masked:
            # Set weights above the diagonal to -inf before the softmax,
            # which will become 0 after the softmax.
            qk_t = qk_t.masked_fill(self.mask[:L, :L] == 0, float("-inf"))

        attention_weights = F.softmax(qk_t, dim=-1)
        wei = attention_weights @ v

        # Concatenate heads and project back to the original embedding dimension.
        wei = wei.reshape(N, L, self.d_v * self.num_heads)
        out = self.fc_out(wei)

        return out
