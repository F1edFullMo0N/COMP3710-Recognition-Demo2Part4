import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation_dataset import OASISSegmentationDataset
from unet import UNet


NUM_CLASSES = 4


def calculate_dice(prediction, target, num_classes=4):
    scores = []

    for class_id in range(num_classes):
        pred_class = prediction == class_id
        true_class = target == class_id

        intersection = (pred_class & true_class).sum().item()
        denominator = (
            pred_class.sum().item()
            + true_class.sum().item()
        )

        dice = (
            2.0 * intersection + 1e-6
        ) / (
            denominator + 1e-6
        )

        scores.append(dice)

    return scores


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
        default="checkpoints/best_unet.pth"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=8
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    image_dir = os.path.join(
        args.data_root,
        "keras_png_slices_test"
    )

    mask_dir = os.path.join(
        args.data_root,
        "keras_png_slices_seg_test"
    )

    dataset = OASISSegmentationDataset(
        image_dir,
        mask_dir
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    print("Test samples:", len(dataset))

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation DSC:",
        checkpoint["dice_scores"]
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    all_intersections = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    all_denominators = torch.zeros(
        NUM_CLASSES,
        dtype=torch.float64
    )

    examples = []

    with torch.no_grad():

        for index, (image, mask) in enumerate(loader):

            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)

            prediction = torch.argmax(
                logits,
                dim=1
            )

            for class_id in range(NUM_CLASSES):

                pred_class = (
                    prediction == class_id
                )

                true_class = (
                    mask == class_id
                )

                all_intersections[class_id] += (
                    pred_class & true_class
                ).sum().cpu()

                all_denominators[class_id] += (
                    pred_class.sum()
                    + true_class.sum()
                ).cpu()

            if len(examples) < args.num_images:
                examples.append(
                    (
                        image[0, 0].cpu().numpy(),
                        mask[0].cpu().numpy(),
                        prediction[0].cpu().numpy()
                    )
                )

    dice_scores = (
        2.0 * all_intersections + 1e-6
    ) / (
        all_denominators + 1e-6
    )

    print("\nTest DSC:")

    for class_id, score in enumerate(dice_scores):
        print(
            f"Class {class_id}: {score:.4f}"
        )

    print(
        f"Mean DSC: {dice_scores.mean():.4f}"
    )

    print(
        f"Minimum DSC: {dice_scores.min():.4f}"
    )

    # ---------------------------------------------
    # Visualise example segmentations
    # ---------------------------------------------

    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(9, 3 * len(examples))
    )

    if len(examples) == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, (image, true_mask, pred_mask) in enumerate(examples):

        axes[row, 0].imshow(
            image,
            cmap="gray"
        )

        axes[row, 0].set_title(
            "MRI"
        )

        axes[row, 1].imshow(
            true_mask,
            vmin=0,
            vmax=3
        )

        axes[row, 1].set_title(
            "Ground Truth"
        )

        axes[row, 2].imshow(
            pred_mask,
            vmin=0,
            vmax=3
        )

        axes[row, 2].set_title(
            "Prediction"
        )

        for column in range(3):
            axes[row, column].axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        args.output_dir,
        "unet_test_predictions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSaved:",
        output_path
    )


if __name__ == "__main__":
    main()