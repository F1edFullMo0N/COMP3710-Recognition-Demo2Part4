import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import OASISDataset
from vae import VAE


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    latent_dim = checkpoint["latent_dim"]

    model = VAE(
        latent_dim=latent_dim
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']}"
    )
    print(
        f"Validation loss: {checkpoint['validation_loss']:.4f}"
    )

    return model


def plot_reconstructions(model, dataset, device, output_dir):
    # Choose several images spread across the test set
    indices = np.linspace(
        0,
        len(dataset) - 1,
        8,
        dtype=int
    )

    images = torch.stack([
        dataset[i]
        for i in indices
    ]).to(device)

    with torch.no_grad():
        mu, logvar = model.encode(images)

        # Use mu rather than random sampling for
        # deterministic reconstruction
        reconstructions = model.decode(mu)

    images = images.cpu()
    reconstructions = reconstructions.cpu()

    fig, axes = plt.subplots(
        2,
        len(indices),
        figsize=(16, 4)
    )

    for i in range(len(indices)):
        axes[0, i].imshow(
            images[i, 0],
            cmap="gray"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstructions[i, 0],
            cmap="gray"
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=12
    )

    axes[1, 0].set_ylabel(
        "Reconstructed",
        fontsize=12
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "vae_reconstructions.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def get_slice_number(filename):
    match = re.search(
        r"slice_(\d+)",
        filename
    )

    if match:
        return int(match.group(1))

    return 0


def plot_latent_space(model, dataset, device, output_dir):
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    all_mu = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            mu, _ = model.encode(images)

            all_mu.append(
                mu.cpu()
            )

    latent = torch.cat(
        all_mu,
        dim=0
    ).numpy()

    slice_numbers = np.array([
        get_slice_number(filename)
        for filename in dataset.image_files
    ])

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        latent[:, 0],
        latent[:, 1],
        c=slice_numbers,
        s=12,
        alpha=0.7
    )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("VAE Latent Space")

    plt.colorbar(
        scatter,
        label="MRI slice number"
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "vae_latent_space.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)

    print(
        "Latent dimension 1 range:",
        latent[:, 0].min(),
        latent[:, 0].max()
    )

    print(
        "Latent dimension 2 range:",
        latent[:, 1].min(),
        latent[:, 1].max()
    )


def plot_manifold(model, device, output_dir):
    grid_size = 15

    # Sample the 2-D latent space
    values = torch.linspace(
        -3.0,
        3.0,
        grid_size
    )

    figure = np.zeros(
        (
            256 * grid_size,
            256 * grid_size
        ),
        dtype=np.float32
    )

    with torch.no_grad():

        for row, z2 in enumerate(
            reversed(values)
        ):
            for col, z1 in enumerate(values):

                z = torch.tensor(
                    [[z1.item(), z2.item()]],
                    dtype=torch.float32,
                    device=device
                )

                generated = model.decode(z)

                image = (
                    generated[0, 0]
                    .cpu()
                    .numpy()
                )

                y1 = row * 256
                y2 = y1 + 256

                x1 = col * 256
                x2 = x1 + 256

                figure[
                    y1:y2,
                    x1:x2
                ] = image

    plt.figure(
        figsize=(12, 12)
    )

    plt.imshow(
        figure,
        cmap="gray",
        extent=[
            -3,
            3,
            -3,
            3
        ]
    )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")

    plt.title(
        "VAE Latent Manifold"
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "vae_manifold.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/groups/comp3710/OASIS"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_vae.pth"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    test_dir = os.path.join(
        args.data_root,
        "keras_png_slices_test"
    )

    test_dataset = OASISDataset(
        test_dir
    )

    print(
        "Test images:",
        len(test_dataset)
    )

    model = load_model(
        args.checkpoint,
        device
    )

    plot_reconstructions(
        model,
        test_dataset,
        device,
        args.output_dir
    )

    plot_latent_space(
        model,
        test_dataset,
        device,
        args.output_dir
    )

    plot_manifold(
        model,
        device,
        args.output_dir
    )


if __name__ == "__main__":
    main()