import torch
import torch.nn as nn

from layers.transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        vocab_size: int,
        context_window_len: int,
        num_layers: int = 6,
        hidden_dim: int | None = None,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.1,
        mask: bool = True,
    ):
        """
        Transformer architecture.
        """
        super(Transformer, self).__init__()
        self.context_window_len = context_window_len

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(context_window_len, embed_dim)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim, num_heads, hidden_dim, d_k, d_v, dropout, mask
                )
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x) + self.positional_embeddings(
            torch.arange(x.shape[1], device=x.device)
        )
        z = self.transformer_blocks(x)
        z = self.ln(z)
        out = self.fc_out(z)
        return out
