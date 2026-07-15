"""
Maps raw model predictions to human-readable severity levels
with clinical descriptions for the demo UI.
"""

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn.functional as F


SEVERITY_LEVELS = ["Mild", "Moderate", "Severe"]

SEVERITY_COLORS = {
    "Mild":     "#22c55e",   # green
    "Moderate": "#f97316",   # orange
    "Severe":   "#ef4444",   # red
}

CLINICAL_DESCRIPTIONS = {
    "tuberculosis": {
        "Mild": (
            "Limited involvement affecting less than 1/3 of one lung. "
            "Early-stage TB with localised infiltrates. "
            "Generally responds well to standard 6-month DOTS therapy."
        ),
        "Moderate": (
            "Involvement of 1/3 to 2/3 of one lung, or smaller bilateral involvement. "
            "Moderate cavitation may be present. "
            "Standard treatment course; monitor for drug resistance."
        ),
        "Severe": (
            "Extensive involvement affecting more than 2/3 of total lung volume, "
            "or significant bilateral disease with cavitation. "
            "Requires aggressive treatment; risk of complications is higher."
        ),
    },
    "pneumonia": {
        "Mild": (
            "Unilobar consolidation with limited extent. "
            "Typically managed with oral antibiotics in outpatient setting. "
            "PSI Class I–II / CURB-65 score 0–1."
        ),
        "Moderate": (
            "Multilobar or bilateral involvement without respiratory failure. "
            "May require hospitalisation and IV antibiotics. "
            "PSI Class III–IV / CURB-65 score 2."
        ),
        "Severe": (
            "Diffuse bilateral infiltrates covering >50% of lung fields. "
            "High risk of respiratory failure; ICU admission often warranted. "
            "PSI Class V / CURB-65 score ≥3."
        ),
    },
}


@dataclass
class SeverityResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    color: str
    description: str
    disclaimer: str = (
        "⚠️ This is an AI-assisted tool for research purposes only. "
        "It is NOT a substitute for clinical diagnosis by a qualified radiologist or physician."
    )


def predict_severity(
    logits: torch.Tensor,
    disease: str = "tuberculosis"
) -> SeverityResult:
    """
    Convert model logits to a SeverityResult.

    Args:
        logits: Raw model output tensor, shape (1, num_classes) or (num_classes,)
        disease: "tuberculosis" or "pneumonia"

    Returns:
        SeverityResult with label, confidence, probabilities, and description
    """
    if logits.dim() == 2:
        logits = logits.squeeze(0)

    probs = F.softmax(logits, dim=0)
    pred_idx = torch.argmax(probs).item()
    label = SEVERITY_LEVELS[pred_idx]
    confidence = probs[pred_idx].item()

    prob_dict = {
        level: round(probs[i].item(), 4)
        for i, level in enumerate(SEVERITY_LEVELS)
    }

    disease_key = disease.lower()
    description = CLINICAL_DESCRIPTIONS.get(disease_key, {}).get(label, "")

    return SeverityResult(
        label=label,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        color=SEVERITY_COLORS[label],
        description=description,
    )


def get_severity_bar(probabilities: Dict[str, float]) -> List[Dict]:
    """Format probabilities for display as confidence bars."""
    return [
        {
            "label": level,
            "probability": probabilities.get(level, 0.0),
            "color": SEVERITY_COLORS[level],
            "percentage": f"{probabilities.get(level, 0.0) * 100:.1f}%",
        }
        for level in SEVERITY_LEVELS
    ]
