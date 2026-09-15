import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OASISSegmentationDataset(Dataset):
    """
    OASIS MRI segmentation dataset.

    Image:
        case_001_slice_10.nii.png

    Mask:
        seg_001_slice_10.nii.png

    Original mask values:
        0, 85, 170, 255

    Converted class labels:
        0, 1, 2, 3
    """

    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

        # Verify every MRI has a corresponding segmentation mask
        for image_file in self.image_files:
            mask_file = self._get_mask_filename(image_file)

            mask_path = os.path.join(
                self.mask_dir,
                mask_file
            )

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask not found for {image_file}: "
                    f"{mask_path}"
                )

    def _get_mask_filename(self, image_filename):
        return image_filename.replace(
            "case_",
            "seg_",
            1
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        mask_filename = self._get_mask_filename(
            image_filename
        )

        image_path = os.path.join(
            self.image_dir,
            image_filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_filename
        )

        # -------------------------------------------------
        # MRI
        # -------------------------------------------------

        image = Image.open(
            image_path
        ).convert("L")

        image = np.array(
            image,
            dtype=np.float32
        )

        # [0, 255] -> [0, 1]
        image /= 255.0

        # [H, W] -> [1, H, W]
        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        # -------------------------------------------------
        # Segmentation mask
        # -------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")

        mask = np.array(
            mask,
            dtype=np.uint8
        )

        # OASIS mask values:
        # 0   -> class 0
        # 85  -> class 1
        # 170 -> class 2
        # 255 -> class 3

        class_mask = np.zeros_like(
            mask,
            dtype=np.int64
        )

        class_mask[mask == 0] = 0
        class_mask[mask == 85] = 1
        class_mask[mask == 170] = 2
        class_mask[mask == 255] = 3

        mask = torch.from_numpy(
            class_mask
        ).long()

        return image, mask