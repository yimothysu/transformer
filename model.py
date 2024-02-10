from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from layers.transformer import Transformer
from utils.tokenizer import Tokenizer


@dataclass
class ModelConfig:
    embed_dim: int
    num_heads: int
    num_layers: int
    context_window_len: int
    hidden_dim: int
    d_k: int | None
    d_v: int | None
    dropout: float
    mask: bool


device = "cuda" if torch.cuda.is_available() else "cpu"


class Model:
    def __init__(self, tokenizer: Tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.transformer = Transformer(
            embed_dim=config.embed_dim,
            vocab_size=tokenizer.vocab_size,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            context_window_len=config.context_window_len,
            hidden_dim=config.hidden_dim,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
            mask=config.mask,
        )
        if torch.cuda.device_count() > 1:
            self.transformer = nn.DataParallel(self.transformer)

    def generate(self, input_text: str, max_len: int = 100, device=device):
        tokens = self.tokenizer.encode(input_text)
        for _ in range(max_len):
            probabilities_matrix = self.transformer(
                torch.Tensor(tokens).to(dtype=torch.int, device=device).unsqueeze(0)
            )
            probabilities_vector = probabilities_matrix[:, -1, :].squeeze(0)
            probabilities_vector = F.softmax(probabilities_vector, dim=0)
            # predicted_token = torch.multinomial(
            #     probabilities_vector, num_samples=1
            # ).item()
            predicted_token = torch.argmax(probabilities_vector).item()
            tokens.append(predicted_token)

        decoded_tokens = self.tokenizer.decode(tokens)
        return decoded_tokens


default_model_config = ModelConfig(
    embed_dim=6 * 64,
    num_heads=6,
    num_layers=6,
    hidden_dim=6 * 64 * 4,
    context_window_len=128,
    dropout=0.2,
    d_k=None,
    d_v=None,
    mask=True,
)
