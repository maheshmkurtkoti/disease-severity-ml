"""
Prepare Pneumonia dataset manifest CSV.

Usage:
    python tasks/pneumonia_severity/prepare_data.py
"""

import random
import pandas as pd
from pathlib import Path

# Expected structure after Kaggle download:
# data/raw/pneumonia/chest_xray/
#   train/NORMAL/*.jpeg
#   train/PNEUMONIA/*.jpeg  (bacterial + viral)

RAW_DIR = "data/raw/pneumonia/chest_xray"
OUTPUT_CSV = "data/processed/pneumonia/manifest.csv"

# Pneumonia cases: distribute into severity levels
# Only PNEUMONIA images get severity labels (NORMAL images are excluded
# since this is severity classification, not detection)
SEVERITY_WEIGHTS = {"Mild": 0.40, "Moderate": 0.40, "Severe": 0.20}


def assign_severity(seed: int) -> str:
    random.seed(seed)
    r = random.random()
    if r < SEVERITY_WEIGHTS["Mild"]:
        return "Mild"
    elif r < SEVERITY_WEIGHTS["Mild"] + SEVERITY_WEIGHTS["Moderate"]:
        return "Moderate"
    else:
        return "Severe"


def prepare_pneumonia_manifest():
    records = []
    raw_path = Path(RAW_DIR)

    for split in ["train", "val", "test"]:
        pneumonia_dir = raw_path / split / "PNEUMONIA"
        if not pneumonia_dir.exists():
            continue

        images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg")) + list(pneumonia_dir.glob("*.png"))
        print(f"  {split}: {len(images)} pneumonia images")

        for i, img_path in enumerate(images):
            records.append({
                "image_path": str(img_path),
                "severity": assign_severity(seed=i),
                "source": "kaggle_chest_xray",
            })

    if not records:
        print("❌ No images found. Download the dataset first.")
        return

    df = pd.DataFrame(records)
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Manifest saved: {output_path}")
    print(f"   Total: {len(df)} | Distribution:\n{df['severity'].value_counts()}")


if __name__ == "__main__":
    prepare_pneumonia_manifest()
