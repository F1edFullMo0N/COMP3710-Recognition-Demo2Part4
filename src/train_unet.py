import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from segmentation_dataset import OASISSegmentationDataset
from unet import UNet


NUM_CLASSES = 4


def dice_loss(logits, targets, num_classes=NUM_CLASSES, smooth=1e-6):
    """
    Soft Dice loss.

    logits:
        [B, C, H, W]

    targets:
        [B, H, W]
        class values 0, 1, 2, 3
    """

    # Convert logits to class probabilities
    probabilities = torch.softmax(logits, dim=1)

    # [B, H, W]
    # ->
    # [B, H, W, C]
    one_hot = F.one_hot(
        targets,
        num_classes=num_classes
    )

    # ->
    # [B, C, H, W]
    one_hot = one_hot.permute(
        0, 3, 1, 2
    ).float()

    # Sum over batch and spatial dimensions
    dimensions = (0, 2, 3)

    intersection = torch.sum(
        probabilities * one_hot,
        dim=dimensions
    )

    denominator = torch.sum(
        probabilities + one_hot,
        dim=dimensions
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    # We want to maximise Dice,
    # so loss = 1 - Dice
    return 1.0 - dice.mean()


def combined_loss(logits, targets, ce_loss_function):
    ce = ce_loss_function(
        logits,
        targets
    )

    dice = dice_loss(
        logits,
        targets
    )

    total = ce + dice

    return total, ce, dice


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    ce_loss_function
):
    model.train()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0

    for images, masks in loader:
        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = model(images)

        loss, ce, d_loss = combined_loss(
            logits,
            masks,
            ce_loss_function
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_ce += ce.item()
        total_dice_loss += d_loss.item()

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_ce / num_batches,
        total_dice_loss / num_batches
    )


def validate(
    model,
    loader,
    device,
    ce_loss_function,
    num_classes=NUM_CLASSES
):
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_dice_loss = 0.0

    # These accumulate statistics over the
    # ENTIRE validation set.
    intersections = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device
    )

    denominators = torch.zeros(
        num_classes,
        dtype=torch.float64,
        device=device
    )

    with torch.no_grad():

        for images, masks in loader:
            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss, ce, d_loss = combined_loss(
                logits,
                masks,
                ce_loss_function
            )

            total_loss += loss.item()
            total_ce += ce.item()
            total_dice_loss += d_loss.item()

            # Hard categorical prediction
            predictions = torch.argmax(
                logits,
                dim=1
            )

            # Calculate each class separately
            for class_id in range(num_classes):

                pred_class = (
                    predictions == class_id
                )

                true_class = (
                    masks == class_id
                )

                intersection = (
                    pred_class & true_class
                ).sum()

                denominator = (
                    pred_class.sum()
                    + true_class.sum()
                )

                intersections[class_id] += intersection
                denominators[class_id] += denominator

    dice_scores = (
        2.0 * intersections + 1e-6
    ) / (
        denominators + 1e-6
    )

    num_batches = len(loader)

    return (
        total_loss / num_batches,
        total_ce / num_batches,
        total_dice_loss / num_batches,
        dice_scores.cpu()
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
        default=30
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
        "--workers",
        type=int,
        default=4
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)

    # ---------------------------------------------
    # Dataset paths
    # ---------------------------------------------

    train_image_dir = os.path.join(
        args.data_root,
        "keras_png_slices_train"
    )

    train_mask_dir = os.path.join(
        args.data_root,
        "keras_png_slices_seg_train"
    )

    val_image_dir = os.path.join(
        args.data_root,
        "keras_png_slices_validate"
    )

    val_mask_dir = os.path.join(
        args.data_root,
        "keras_png_slices_seg_validate"
    )

    # ---------------------------------------------
    # Datasets
    # ---------------------------------------------

    train_dataset = OASISSegmentationDataset(
        train_image_dir,
        train_mask_dir
    )

    val_dataset = OASISSegmentationDataset(
        val_image_dir,
        val_mask_dir
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # ---------------------------------------------
    # DataLoaders
    # ---------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )

    # ---------------------------------------------
    # Model
    # ---------------------------------------------

    model = UNet(
        in_channels=1,
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    ce_loss_function = nn.CrossEntropyLoss()

    os.makedirs(
        "checkpoints",
        exist_ok=True
    )

    # Since the assignment requires ALL classes
    # to exceed 0.9 DSC, optimise checkpoint
    # selection according to the worst class.
    best_min_dice = -1.0

    # ---------------------------------------------
    # Training
    # ---------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_ce, train_dice_loss = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                ce_loss_function
            )
        )

        (
            val_loss,
            val_ce,
            val_dice_loss,
            val_dice_scores
        ) = validate(
            model,
            val_loader,
            device,
            ce_loss_function
        )

        mean_dice = (
            val_dice_scores.mean().item()
        )

        min_dice = (
            val_dice_scores.min().item()
        )

        dice_text = " | ".join(
            [
                f"Class {i}: {score:.4f}"
                for i, score
                in enumerate(val_dice_scores)
            ]
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} "
            f"(CE {train_ce:.4f}, "
            f"DiceLoss {train_dice_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Validation DSC | "
            f"{dice_text} | "
            f"Mean: {mean_dice:.4f} | "
            f"Min: {min_dice:.4f}"
        )

        # -----------------------------------------
        # Save best model based on WORST class DSC
        # -----------------------------------------

        if min_dice > best_min_dice:
            best_min_dice = min_dice

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_loss":
                        val_loss,
                    "dice_scores":
                        val_dice_scores,
                    "mean_dice":
                        mean_dice,
                    "min_dice":
                        min_dice,
                    "num_classes":
                        NUM_CLASSES
                },
                "checkpoints/best_unet.pth"
            )

            print(
                "Saved new best model "
                f"(minimum DSC = {min_dice:.4f})"
            )

        if min_dice > 0.90:
            print(
                "*** All validation classes "
                "are above 0.90 DSC! ***"
            )

    print("Training finished.")
    print(
        "Best minimum validation DSC:",
        best_min_dice
    )


if __name__ == "__main__":
    main()