import argparse

import torch

# These imports prevent "AttributeError: Can't get attribute 'Tokenizer' on <module '__main__' ...>"
from utils.tokenizer import Tokenizer
from layers.transformer import Transformer
from layers.transformer_block import TransformerBlock
from layers.self_attention import SelfAttention
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="models/model.pt", help="Path to the model"
    )
    parser.add_argument("--input_text", type=str, help="Input text for generation")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive generation"
    )
    parser.add_argument(
        "--max_len", type=int, default=100, help="Maximum length for generation"
    )
    args = parser.parse_args()

    if not args.input_text and not args.interactive:
        raise ValueError("Either input_text or interactive should be set")

    model: Model = torch.load(args.model_path)
    model.transformer.to(device=device)
    if args.interactive:
        while True:
            input_text = input("Enter a Prompt: ")
            output = model.generate(
                input_text=input_text, max_len=args.max_len, device=device
            )
            print("Model Response:", output, "\n")
    else:
        output = model.generate(
            input_text=args.input_text, max_len=args.max_len, device=device
        )
        print(output)
