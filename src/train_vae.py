import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import OASISDataset
from vae import VAE


def vae_loss(reconstruction, x, mu, logvar):
    # Reconstruction loss:
    # How similar is the reconstructed MRI to the original MRI?
    reconstruction_loss = F.mse_loss(
        reconstruction,
        x,
        reduction="sum"
    )

    # KL divergence:
    # Encourage the latent distribution to stay close to N(0, 1)
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    # Average over the batch
    batch_size = x.size(0)

    reconstruction_loss /= batch_size
    kl_loss /= batch_size

    total_loss = reconstruction_loss + kl_loss

    return total_loss, reconstruction_loss, kl_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    for images in loader:
        images = images.to(device)

        optimizer.zero_grad()

        reconstruction, mu, logvar = model(images)

        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches
    )


def validate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_kl = 0.0

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            reconstruction, mu, logvar = model(images)

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                images,
                mu,
                logvar
            )

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_reconstruction / num_batches,
        total_kl / num_batches
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/groups/comp3710/OASIS"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--latent_dim",
        type=int,
        default=2
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_dir = os.path.join(
        args.data_root,
        "keras_png_slices_train"
    )

    validation_dir = os.path.join(
        args.data_root,
        "keras_png_slices_validate"
    )

    train_dataset = OASISDataset(train_dir)
    validation_dataset = OASISDataset(validation_dir)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(validation_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = VAE(
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------

    os.makedirs("checkpoints", exist_ok=True)

    best_validation_loss = float("inf")

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        train_loss, train_recon, train_kl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss, val_recon, val_kl = validate(
            model,
            validation_loader,
            device
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train: {train_loss:.4f} "
            f"(Recon {train_recon:.4f}, KL {train_kl:.4f}) | "
            f"Val: {val_loss:.4f} "
            f"(Recon {val_recon:.4f}, KL {val_kl:.4f})"
        )

        # Save the model with the best validation loss
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_loss": val_loss,
                    "latent_dim": args.latent_dim
                },
                "checkpoints/best_vae.pth"
            )

            print("Saved new best model.")

    print("Training finished.")
    print(
        "Best validation loss:",
        best_validation_loss
    )


if __name__ == "__main__":
    main()