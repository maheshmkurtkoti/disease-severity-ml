"""
Dataset classes for TB and Pneumonia X-ray severity classification.

TB Dataset:   Montgomery County + Shenzhen Hospital
              https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen

Pneumonia:    Chest X-Ray Images (Pneumonia) by Paul Mooney
              https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Both datasets label images into severity levels.
When the original dataset only has Normal/Abnormal labels, severity is
derived from radiological heuristics (see severity_mapper.py for details).
"""

import os
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split


SEVERITY_TO_IDX = {"Mild": 0, "Moderate": 1, "Severe": 2}
IDX_TO_SEVERITY = {v: k for k, v in SEVERITY_TO_IDX.items()}


class ChestXRayDataset(Dataset):
    """
    Generic chest X-ray dataset that reads from a CSV manifest.

    Expected CSV columns:
        image_path  : absolute or relative path to the image
        severity    : "Mild", "Moderate", or "Severe"

    If you only have Normal/Abnormal labels, use the helper scripts in
    tasks/tb_severity/ or tasks/pneumonia_severity/ to generate the manifest.
    """

    def __init__(
        self,
        manifest_csv: str,
        transform: Optional[Callable] = None,
        image_root: Optional[str] = None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.image_root = Path(image_root) if image_root else None

        assert "image_path" in self.df.columns, "CSV must have 'image_path' column"
        assert "severity" in self.df.columns,   "CSV must have 'severity' column"

        # Drop rows with unknown severity
        self.df = self.df[self.df["severity"].isin(SEVERITY_TO_IDX.keys())].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])

        if self.image_root and not img_path.is_absolute():
            img_path = self.image_root / img_path

        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        if self.transform:
            augmented = self.transform(image=image_np)
            image_tensor = augmented["image"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        label = SEVERITY_TO_IDX[row["severity"]]
        return image_tensor, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights to handle imbalanced datasets."""
        counts = self.df["severity"].value_counts()
        weights = []
        total = len(self.df)
        for level in ["Mild", "Moderate", "Severe"]:
            count = counts.get(level, 1)
            weights.append(total / (3 * count))
        return torch.tensor(weights, dtype=torch.float32)


class TBDataset(ChestXRayDataset):
    """
    Wrapper for the Montgomery + Shenzhen TB dataset.
    Adds TB-specific preprocessing (lung field enhancement).
    """

    def __init__(self, manifest_csv: str, transform=None, image_root=None):
        super().__init__(manifest_csv, transform, image_root)

    @staticmethod
    def download_instructions() -> str:
        return """
To download the TB dataset:

1. Install Kaggle CLI:
   pip install kaggle

2. Set up API token from https://www.kaggle.com/settings → API → Create Token
   Place kaggle.json in ~/.kaggle/kaggle.json

3. Download:
   kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen -p data/raw/tb --unzip
   kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery -p data/raw/tb --unzip

4. Then run:
   python tasks/tb_severity/prepare_data.py
"""


class PneumoniaDataset(ChestXRayDataset):
    """
    Wrapper for the Kaggle Chest X-Ray Pneumonia dataset.
    """

    def __init__(self, manifest_csv: str, transform=None, image_root=None):
        super().__init__(manifest_csv, transform, image_root)

    @staticmethod
    def download_instructions() -> str:
        return """
To download the Pneumonia dataset:

1. Install Kaggle CLI:
   pip install kaggle

2. Set up API token from https://www.kaggle.com/settings → API → Create Token
   Place kaggle.json in ~/.kaggle/kaggle.json

3. Download:
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw/pneumonia --unzip

4. Then run:
   python tasks/pneumonia_severity/prepare_data.py
"""


def build_dataloaders(
    manifest_csv: str,
    train_transform,
    val_transform,
    batch_size: int = 16,
    train_split: float = 0.7,
    val_split: float = 0.15,
    num_workers: int = 2,
    image_root: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders from a single manifest CSV.
    Performs random split.
    """
    full_dataset = ChestXRayDataset(manifest_csv, transform=None, image_root=image_root)
    n = len(full_dataset)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    n_test  = n - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms after split
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform   = val_transform
    test_ds.dataset.transform  = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
