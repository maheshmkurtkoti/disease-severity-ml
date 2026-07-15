"""
Run TB severity model training.

Usage:
    python tasks/tb_severity/run_training.py
    python tasks/tb_severity/run_training.py --config configs/tb.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml
import torch

from src.data.dataset_factory import build_dataloaders, ChestXRayDataset
from src.models.cnn_backbone import build_model
from src.training.trainer import Trainer
from src.utils.transforms import get_train_transforms, get_val_transforms
from src.evaluation.metrics import evaluate_model, print_metrics
from src.models.cnn_backbone import load_model


def main(config_path: str = "configs/tb.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    manifest_csv = Path(config["paths"]["processed_dir"]) / "manifest.csv"
    if not manifest_csv.exists():
        print(f"❌ Manifest not found at {manifest_csv}")
        print("   Run: python tasks/tb_severity/prepare_data.py")
        return

    # Build transforms
    train_tf = get_train_transforms(config["data"]["image_size"])
    val_tf   = get_val_transforms(config["data"]["image_size"])

    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        manifest_csv=str(manifest_csv),
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=config["training"]["batch_size"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
    )

    # Get class weights for imbalanced data
    full_ds = ChestXRayDataset(str(manifest_csv))
    class_weights = full_ds.get_class_weights()
    print(f"Class weights: {class_weights}")

    # Build and train model
    model = build_model(config)
    print(f"Trainable params: {model.get_trainable_params():,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        class_weights=class_weights,
    )

    history = trainer.train()

    # Evaluate on test set
    print("\n📊 Test Set Evaluation:")
    best_model_path = Path(config["paths"]["model_save_dir"]) / "tuberculosis_best.pth"
    if best_model_path.exists():
        best_model, _ = load_model(str(best_model_path), device=device)
        metrics = evaluate_model(best_model, test_loader, device=device)
        print_metrics(metrics, disease="tuberculosis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tb.yaml")
    args = parser.parse_args()
    main(args.config)
