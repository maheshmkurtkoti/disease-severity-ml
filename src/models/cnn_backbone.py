"""
EfficientNet-B0 backbone for severity classification.

Why EfficientNet-B0?
- Achieves strong accuracy with low parameter count (~5.3M)
- Runs well on CPU — no GPU required for inference
- Pretrained on ImageNet transfers well to medical imaging
- Easily upgradeable to B1-B7 for more accuracy if needed
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path


class SeverityClassifier(nn.Module):
    """
    EfficientNet-B0 with a custom classification head for severity prediction.
    Outputs logits for [Mild, Moderate, Severe].
    """

    def __init__(
        self,
        num_classes: int = 3,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained EfficientNet, remove its original classifier
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,          # Remove head — we add our own
            global_pool="avg",
        )

        in_features = self.backbone.num_features  # 1280 for B0

        # Custom head: BatchNorm → Dropout → Linear
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze backbone weights — useful for first few epochs of fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all weights for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> SeverityClassifier:
    """Build model from config dict."""
    return SeverityClassifier(
        num_classes=config["data"]["num_classes"],
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    )


def load_model(checkpoint_path: str, device: str = "cpu") -> tuple[SeverityClassifier, dict]:
    """
    Load a saved model checkpoint.

    Returns:
        (model, metadata) where metadata includes disease, class_names, epoch, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = SeverityClassifier(
        num_classes=checkpoint.get("num_classes", 3),
        backbone=checkpoint.get("backbone", "efficientnet_b0"),
        pretrained=False,  # We're loading weights, not downloading
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {
        "disease":      checkpoint.get("disease", "unknown"),
        "class_names":  checkpoint.get("class_names", ["Mild", "Moderate", "Severe"]),
        "epoch":        checkpoint.get("epoch", 0),
        "val_accuracy": checkpoint.get("val_accuracy", None),
    }

    return model, metadata


def save_model(
    model: SeverityClassifier,
    path: str,
    disease: str,
    epoch: int,
    val_accuracy: float,
    class_names: list,
):
    """Save model checkpoint with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes":      model.num_classes,
        "backbone":         "efficientnet_b0",
        "dropout":          0.3,
        "disease":          disease,
        "class_names":      class_names,
        "epoch":            epoch,
        "val_accuracy":     val_accuracy,
    }, path)
    print(f"✅ Model saved to {path}")
