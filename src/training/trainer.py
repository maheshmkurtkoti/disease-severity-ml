"""
Training loop with:
- Cosine LR scheduling
- Early stopping
- Backbone freeze/unfreeze strategy
- Weighted loss for class imbalance
- Checkpoint saving
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.cnn_backbone import SeverityClassifier, save_model
from src.evaluation.metrics import compute_metrics, print_metrics


class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: SeverityClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss: weighted cross-entropy handles class imbalance
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
        )

        self.early_stopping = EarlyStopping(
            patience=config["training"]["early_stopping_patience"]
        )

        self.best_val_accuracy = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    def train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc="  Training", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    @torch.no_grad()
    def validate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc="  Validating", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        metrics = compute_metrics(all_labels, all_preds)
        return avg_loss, metrics["accuracy"]

    def train(self):
        cfg = self.config
        epochs = cfg["training"]["epochs"]
        freeze_epochs = cfg["model"]["freeze_backbone_epochs"]
        disease = cfg["disease"]
        save_dir = Path(cfg["paths"]["model_save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🚀 Starting training: {disease.upper()}")
        print(f"   Epochs: {epochs} | Device: {self.device}")
        print(f"   Backbone frozen for first {freeze_epochs} epochs\n")

        # Freeze backbone initially
        self.model.freeze_backbone()

        for epoch in range(1, epochs + 1):
            start = time.time()

            # Unfreeze backbone after warmup
            if epoch == freeze_epochs + 1:
                print("🔓 Unfreezing backbone for full fine-tuning")
                self.model.unfreeze_backbone()
                # Re-init optimizer to include backbone params
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=cfg["training"]["learning_rate"] * 0.1,
                    weight_decay=cfg["training"]["weight_decay"],
                )

            train_loss = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            elapsed = time.time() - start
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)

            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                save_model(
                    model=self.model,
                    path=str(save_dir / f"{disease}_best.pth"),
                    disease=disease,
                    epoch=epoch,
                    val_accuracy=val_acc,
                    class_names=cfg["data"]["class_names"],
                )
                print(f"  ⭐ New best model saved (val_acc={val_acc:.4f})")

            if self.early_stopping.step(val_acc):
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

        print(f"\n✅ Training complete. Best val accuracy: {self.best_val_accuracy:.4f}")
        return self.history
