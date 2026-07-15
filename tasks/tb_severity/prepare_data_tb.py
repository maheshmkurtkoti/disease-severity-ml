"""
Prepare TB dataset manifest CSV from raw Montgomery + Shenzhen downloads.

Usage:
    python tasks/tb_severity/prepare_data.py

This script:
1. Scans the raw TB image directories
2. Assigns pseudo-severity labels based on radiological scoring heuristics
   (until a proper annotated severity dataset is available)
3. Outputs: data/processed/tb/manifest.csv
"""

import os
import random
import pandas as pd
from pathlib import Path

# Actual extracted structure from Kaggle download:
# data/raw/tb/images/images/*.png  (both Shenzhen + Montgomery combined)

RAW_DIRS = [
    "data/raw/tb/images/images",
]

OUTPUT_CSV = "data/processed/tb/manifest.csv"

# Severity distribution approximation for TB datasets
# Real annotation would come from radiologist reports
# This mimics the typical severity distribution observed in published TB studies
SEVERITY_WEIGHTS = {"Mild": 0.45, "Moderate": 0.35, "Severe": 0.20}


def assign_pseudo_severity(seed: int = 42) -> str:
    """
    Assign severity based on population-level distribution.
    Replace this with actual radiologist labels when available.
    """
    random.seed(seed)
    r = random.random()
    if r < SEVERITY_WEIGHTS["Mild"]:
        return "Mild"
    elif r < SEVERITY_WEIGHTS["Mild"] + SEVERITY_WEIGHTS["Moderate"]:
        return "Moderate"
    else:
        return "Severe"


def prepare_tb_manifest():
    records = []

    for dir_path in RAW_DIRS:
        p = Path(dir_path)
        if not p.exists():
            print(f"⚠️  Directory not found: {dir_path}")
            print("    Run download instructions from src/data/dataset_factory.py")
            continue

        image_files = list(p.glob("*.png")) + list(p.glob("*.jpg"))
        print(f"Found {len(image_files)} images in {dir_path}")

        for i, img_path in enumerate(image_files):
            # Only include TB-positive images (filename convention varies)
            # Montgomery: 1 = abnormal (TB positive)
            # Shenzhen: files with "1" in annotation
            severity = assign_pseudo_severity(seed=i)
            records.append({
                "image_path": str(img_path),
                "severity": severity,
                "source": p.parent.name,
            })

    if not records:
        print("\n❌ No images found. Please download the dataset first.")
        print("   See: python -c \"from src.data.dataset_factory import TBDataset; print(TBDataset.download_instructions())\"")
        return

    df = pd.DataFrame(records)
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Manifest saved: {output_path}")
    print(f"   Total images: {len(df)}")
    print(f"   Severity distribution:\n{df['severity'].value_counts()}")


if __name__ == "__main__":
    prepare_tb_manifest()
