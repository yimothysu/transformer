import torch
import torch.nn as nn

from layers.self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.1,
        mask: bool = True,
    ):
        """
        Single block of the transformer architecture.
        """
        super(TransformerBlock, self).__init__()

        if hidden_dim is None:
            hidden_dim = embed_dim

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.sa = SelfAttention(embed_dim, num_heads, d_k, d_v, dropout, mask)

        self.fc_1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc_2 = nn.Linear(hidden_dim, embed_dim)
        self.ffn = nn.Sequential(self.fc_1, self.act, self.fc_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
