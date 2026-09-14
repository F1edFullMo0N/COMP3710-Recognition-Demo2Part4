import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class OASISDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".png")
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        filename = self.image_files[index]
        image_path = os.path.join(self.image_dir, filename)

        # OASIS images are grayscale PNGs
        image = Image.open(image_path).convert("L")

        # Convert to numpy and normalize from [0, 255] to [0, 1]
        image = np.array(image, dtype=np.float32) / 255.0

        # Convert:
        # [H, W] -> [1, H, W]
        image = torch.from_numpy(image).unsqueeze(0)

        return image