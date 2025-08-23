import argparse
from pathlib import Path
from typing import List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TabularAutoencoder(nn.Module):
    """Simple feed-forward autoencoder for tabular data."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def train_autoencoder(
    data: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 32,
    latent_dim: int = 8,
    hidden_dim: int = 64,
) -> TabularAutoencoder:
    """Train autoencoder on given data."""

    feature_cols: List[str] = [
        c for c in data.columns if c.startswith("vle_") or c.startswith("assessment_")
    ]
    x = data[feature_cols].fillna(0).astype("float32").values
    tensor_x = torch.tensor(x)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TabularAutoencoder(x.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def export_latent_vectors(
    model: TabularAutoencoder,
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate latent features and save to parquet."""

    feature_cols: List[str] = [
        c for c in data.columns if c.startswith("vle_") or c.startswith("assessment_")
    ]
    x = data[feature_cols].fillna(0).astype("float32").values
    with torch.no_grad():
        latents = model.encoder(torch.tensor(x)).numpy()

    latent_cols = [f"autoenc_feat{i+1}" for i in range(latents.shape[1])]
    latent_df = pd.DataFrame(latents, columns=latent_cols)
    latent_df["id_student"] = data["id_student"].values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_df.to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train autoencoder on OULAD features")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to processed dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to save latent features")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--latent-dim", type=int, default=8, help="Latent space dimension")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    model = train_autoencoder(
        df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )
    export_latent_vectors(model, df, args.output)


if __name__ == "__main__":
    main()
