import argparse
from dataclasses import dataclass
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch import nn

from tqdm import tqdm

from model import Model, default_model_config
from utils.data import build_datasets_and_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    block_size: int


default_train_config = TrainConfig(
    batch_size=2048,
    learning_rate=3e-4,
    epochs=2,
    block_size=64,
)


def train(model: Model, dataset: Dataset, config: TrainConfig):
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    optimizer = AdamW(model.transformer.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        i = 0
        epoch_loss = 0
        for X, y in tqdm(dataloader):
            optimizer.zero_grad()
            y_pred = model.transformer(X)

            loss = loss_fn(y_pred.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            i += 1
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} / {config.epochs} - Average Loss: {epoch_loss / i}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data", help="Directory containing the dataset"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory containing the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory containing the output",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=default_train_config.batch_size,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=default_train_config.learning_rate,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_train_config.epochs,
        help="Number of epochs",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=default_train_config.block_size,
        help="Block size for training",
    )
    args = parser.parse_args()

    train_config = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        block_size=args.block_size,
    )

    print(f"Data directory: {args.data_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, test_ds, tokenizer = build_datasets_and_tokenizer(
        args.data_dir, train_config.block_size, device=device
    )
    model = Model(tokenizer, default_model_config)
    model.transformer.to(device=device)
    train(model, train_ds, train_config)
    model_save_path = os.path.join(args.model_dir, "model.pt")
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")

    while True:
        input_text = input("Enter a prompt: ")
        output_text = model.generate(input_text)
        print(f"Completion: {output_text}")
        print()
