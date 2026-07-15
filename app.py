"""
Gradio Demo App — Chest X-Ray Severity Classifier
Supports: Tuberculosis | Pneumonia

Run:
    python app.py

Then open http://localhost:7860 in your browser.
To get a public shareable link: python app.py --share
"""

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.models.cnn_backbone import SeverityClassifier, load_model
from src.utils.transforms import get_inference_transforms, pil_to_numpy
from src.utils.severity_mapper import predict_severity, get_severity_bar, SEVERITY_LEVELS

# ─── Model paths ──────────────────────────────────────────────────────────────
MODEL_PATHS = {
    "Tuberculosis": "models/tb/tuberculosis_best.pth",
    "Pneumonia":    "models/pneumonia/pneumonia_best.pth",
}

DISEASE_KEYS = {
    "Tuberculosis": "tuberculosis",
    "Pneumonia":    "pneumonia",
}

EXAMPLE_IMAGES_DIR = Path("examples")

# ─── Load models ──────────────────────────────────────────────────────────────
_models = {}
_transform = get_inference_transforms(image_size=224)


def _get_model(disease_label: str):
    """Lazy-load model on first request."""
    if disease_label in _models:
        return _models[disease_label]

    path = MODEL_PATHS[disease_label]
    if not Path(path).exists():
        return None  # Demo mode — model not trained yet

    model, _ = load_model(path, device="cpu")
    _models[disease_label] = model
    return model


def _dummy_predict(disease_label: str) -> torch.Tensor:
    """
    Return random logits when no model is available.
    Used for demo/UI testing before training.
    """
    torch.manual_seed(42)
    return torch.randn(3)


def run_inference(image: Image.Image, disease_label: str):
    """
    Run severity prediction on a PIL image.
    Returns the severity result.
    """
    image_np = pil_to_numpy(image)
    augmented = _transform(image=image_np)
    tensor = augmented["image"].unsqueeze(0)  # Add batch dim

    model = _get_model(disease_label)

    if model is None:
        # Demo mode: no trained model yet — return mock result
        logits = _dummy_predict(disease_label)
        demo_note = True
    else:
        with torch.no_grad():
            logits = model(tensor).squeeze(0)
        demo_note = False

    disease_key = DISEASE_KEYS[disease_label]
    result = predict_severity(logits, disease=disease_key)

    return result, demo_note


# ─── Gradio inference function ────────────────────────────────────────────────
def predict(image, disease_choice):
    if image is None:
        return (
            "⬆️ Please upload a chest X-ray image",
            {},
            "",
            "",
        )

    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    result, is_demo = run_inference(pil_image, disease_choice)

    # Severity label + confidence
    severity_text = f"**{result.label}**  ({result.confidence * 100:.1f}% confidence)"
    if is_demo:
        severity_text += "\n\n⚠️ *Demo mode: Model not trained yet. Showing mock prediction.*"

    # Probability bar chart data for Gradio
    prob_data = {
        level: result.probabilities.get(level, 0.0)
        for level in SEVERITY_LEVELS
    }

    # Clinical description
    description = result.description

    # Disclaimer
    disclaimer = result.disclaimer

    return severity_text, prob_data, description, disclaimer


# ─── Build Gradio UI ──────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="Chest X-Ray Severity Classifier") as demo:

        gr.Markdown("""
        # 🫁 Chest X-Ray Severity Classifier
        **Tuberculosis & Pneumonia — Mild / Moderate / Severe**

        Upload a chest X-ray and select the disease type to get a severity prediction.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                    height=300,
                )
                disease_input = gr.Radio(
                    choices=["Tuberculosis", "Pneumonia"],
                    value="Tuberculosis",
                    label="Disease Type",
                )
                submit_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")

            with gr.Column(scale=1):
                severity_output = gr.Markdown(
                    label="Severity",
                    value="Upload an X-ray and click Analyse"
                )
                prob_output = gr.Label(
                    label="Confidence by Severity Level",
                    num_top_classes=3,
                )
                description_output = gr.Textbox(
                    label="Clinical Context",
                    lines=4,
                    interactive=False,
                )
                disclaimer_output = gr.Markdown(elem_classes=["disclaimer"])

        submit_btn.click(
            fn=predict,
            inputs=[image_input, disease_input],
            outputs=[severity_output, prob_output, description_output, disclaimer_output],
        )

        gr.Markdown("""
        ---
        ### ℹ️ About
        - **Model**: EfficientNet-B0 fine-tuned on chest X-ray datasets
        - **TB Dataset**: Montgomery County + Shenzhen Hospital X-rays
        - **Pneumonia Dataset**: Kaggle Chest X-Ray Images (Paul Mooney)
        - **Classes**: Mild · Moderate · Severe
        - **Framework**: PyTorch + timm + Gradio

        > This tool is for **research and educational purposes only**.
        > It is not validated for clinical use. Always consult a qualified physician.
        """)

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    ui = build_ui()
    ui.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .severity-card { border-radius: 12px; padding: 16px; }
        .disclaimer { font-size: 0.85em; color: #888; border-left: 3px solid #f97316; padding-left: 10px; }
        """,
    )