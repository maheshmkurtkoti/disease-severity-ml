"""
run_all.py — One command to download data, prepare, and train both models.

Usage:
    python run_all.py

Leave this running overnight. Both models will be trained and saved to:
    models/tb/tuberculosis_best.pth
    models/pneumonia/pneumonia_best.pth
"""

import subprocess
import sys
import time
from pathlib import Path


def run(cmd, description):
    print(f"\n{'='*60}")
    print(f"▶  {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Failed: {description}")
        print(f"   Command was: {cmd}")
        sys.exit(1)
    print(f"✅ Done: {description}")


def main():
    start = time.time()

    print("""
╔══════════════════════════════════════════════════════════╗
║       Disease Severity ML — Overnight Training           ║
║       TB + Pneumonia   |   EfficientNet-B0               ║
╚══════════════════════════════════════════════════════════╝

This will:
  1. Download TB datasets (~400 MB)
  2. Download Pneumonia dataset (~1.2 GB)
  3. Prepare data manifests
  4. Train TB severity model
  5. Train Pneumonia severity model

Estimated time on CPU: 8–12 hours
Go to sleep — it will be done by morning! 🌙
""")

    # ── Create directories ────────────────────────────────────
    for d in [
        "data/raw/tb",
        "data/raw/pneumonia",
        "data/processed/tb",
        "data/processed/pneumonia",
        "models/tb",
        "models/pneumonia",
        "logs",
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── Download TB datasets ──────────────────────────────────
    run(
        "kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen "
        "-p data/raw/tb --unzip",
        "Downloading Shenzhen TB X-rays"
    )

    run(
        "kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery "
        "-p data/raw/tb --unzip",
        "Downloading Montgomery TB X-rays"
    )

    # ── Download Pneumonia dataset ────────────────────────────
    run(
        "kaggle datasets download -d paultimothymooney/chest-xray-pneumonia "
        "-p data/raw/pneumonia --unzip",
        "Downloading Chest X-Ray Pneumonia dataset"
    )

    # ── Prepare data manifests ────────────────────────────────
    run(
        "python tasks/tb_severity/prepare_data.py",
        "Preparing TB data manifest"
    )

    run(
        "python tasks/pneumonia_severity/prepare_data.py",
        "Preparing Pneumonia data manifest"
    )

    # ── Train models ──────────────────────────────────────────
    run(
        "python tasks/tb_severity/run_training.py",
        "Training TB Severity Model (this will take a few hours)"
    )

    run(
        "python tasks/pneumonia_severity/run_training.py",
        "Training Pneumonia Severity Model (this will take a few hours)"
    )

    # ── Done ──────────────────────────────────────────────────
    elapsed = time.time() - start
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║                  🎉 ALL DONE!                            ║
╚══════════════════════════════════════════════════════════╝

  Total time: {hours}h {minutes}m

  Saved models:
    ✅ models/tb/tuberculosis_best.pth
    ✅ models/pneumonia/pneumonia_best.pth

  Now run the demo:
    python app.py
    → Open http://localhost:7860
""")


if __name__ == "__main__":
    main()