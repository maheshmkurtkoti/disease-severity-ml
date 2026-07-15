"""
Evaluation metrics for severity classification.
Reports accuracy, per-class F1, confusion matrix, and quadratic weighted kappa
(the latter is standard in medical ordinal classification).
"""

from typing import Dict, List
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


SEVERITY_LEVELS = ["Mild", "Moderate", "Severe"]


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str] = SEVERITY_LEVELS,
) -> Dict:
    """
    Compute all metrics given true and predicted class indices.

    Returns a dict with:
        accuracy, macro_f1, weighted_f1, per_class_f1, kappa, confusion_matrix
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    return {
        "accuracy":       round(acc, 4),
        "macro_f1":       round(macro_f1, 4),
        "weighted_f1":    round(weighted_f1, 4),
        "per_class_f1":   {class_names[i]: round(float(per_class_f1[i]), 4) for i in range(len(class_names))},
        "kappa":          round(kappa, 4),
        "confusion_matrix": cm.tolist(),
        "report":         report,
    }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    class_names: List[str] = SEVERITY_LEVELS,
) -> Dict:
    """Run evaluation loop and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return compute_metrics(all_labels, all_preds, class_names)


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str] = SEVERITY_LEVELS,
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> plt.Figure:
    """Plot and optionally save a confusion matrix heatmap."""
    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)

    return fig


def print_metrics(metrics: Dict, disease: str = ""):
    prefix = f"[{disease.upper()}] " if disease else ""
    print(f"\n{'='*50}")
    print(f"{prefix}Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Quadratic Kappa: {metrics['kappa']:.4f}")
    print(f"\n  Per-Class F1:")
    for cls, score in metrics["per_class_f1"].items():
        print(f"    {cls:10s}: {score:.4f}")
    print(f"\n{metrics['report']}")
