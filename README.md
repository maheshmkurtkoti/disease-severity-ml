# 🫁 Disease Severity ML — Chest X-Ray Severity Classifier

Classify the severity of **Tuberculosis** and **Pneumonia** from chest X-rays using deep learning.

**Severity levels:** Mild · Moderate · Severe

---

## Demo

```
python app.py
```

Open http://localhost:7860 — upload a chest X-ray, select the disease, get a severity prediction.

To generate a public shareable link (no deployment needed):
```
python app.py --share
```

---

## How It Works

| Component | Detail |
|---|---|
| Model | EfficientNet-B0 (pretrained on ImageNet, fine-tuned on X-rays) |
| Input | Chest X-ray image (any format: PNG, JPG, JPEG) |
| Output | Severity class (Mild/Moderate/Severe) + confidence scores |
| Framework | PyTorch + timm + Gradio |

**Severity criteria:**

_Tuberculosis_ — based on lung field involvement:
- **Mild**: < 1/3 of one lung affected
- **Moderate**: 1/3–2/3 of one lung, or bilateral < 1/3
- **Severe**: > 2/3 total lung volume, significant cavitation

_Pneumonia_ — based on consolidation extent:
- **Mild**: Unilobar involvement (PSI Class I–II)
- **Moderate**: Multilobar or bilateral (PSI Class III–IV)
- **Severe**: Diffuse bilateral > 50% (PSI Class V)

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/maheshmkurtkoti/disease-severity-ml.git
cd disease-severity-ml

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download datasets

You need a free Kaggle account and API token.

**Get your Kaggle token:**
1. Go to https://www.kaggle.com/settings → API → **Create New Token**
2. Save `kaggle.json` to `~/.kaggle/kaggle.json`
3. `chmod 600 ~/.kaggle/kaggle.json`

**Download TB dataset (Montgomery + Shenzhen):**
```bash
kaggle datasets download -d raddar/tuberculosis-chest-xrays-shenzhen -p data/raw/tb --unzip
kaggle datasets download -d raddar/tuberculosis-chest-xrays-montgomery -p data/raw/tb --unzip
```

**Download Pneumonia dataset:**
```bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/raw/pneumonia --unzip
```

### 3. Prepare data

```bash
python tasks/tb_severity/prepare_data.py
python tasks/pneumonia_severity/prepare_data.py
```

This scans the raw images and creates CSV manifests at:
- `data/processed/tb/manifest.csv`
- `data/processed/pneumonia/manifest.csv`

### 4. Train models

```bash
# Train TB severity model
python tasks/tb_severity/run_training.py

# Train Pneumonia severity model
python tasks/pneumonia_severity/run_training.py
```

Trained models are saved to `models/tb/` and `models/pneumonia/`.

### 5. Run the demo

```bash
python app.py
```

---

## Project Structure

```
disease-severity-ml/
│
├── app.py                              # Gradio demo app ← START HERE
├── requirements.txt
│
├── configs/
│   ├── tb.yaml                         # TB training config
│   └── pneumonia.yaml                  # Pneumonia training config
│
├── src/
│   ├── data/
│   │   └── dataset_factory.py          # Dataset classes + dataloader builder
│   ├── models/
│   │   └── cnn_backbone.py             # EfficientNet-B0 classifier
│   ├── training/
│   │   └── trainer.py                  # Training loop, early stopping
│   ├── evaluation/
│   │   └── metrics.py                  # Accuracy, F1, kappa, confusion matrix
│   └── utils/
│       ├── transforms.py               # Augmentation pipelines
│       └── severity_mapper.py          # Logits → severity label + description
│
├── tasks/
│   ├── tb_severity/
│   │   ├── prepare_data.py             # Generate TB manifest CSV
│   │   └── run_training.py             # Train TB model
│   └── pneumonia_severity/
│       ├── prepare_data.py             # Generate Pneumonia manifest CSV
│       └── run_training.py             # Train Pneumonia model
│
├── models/                             # Saved checkpoints (created after training)
│   ├── tb/tuberculosis_best.pth
│   └── pneumonia/pneumonia_best.pth
│
├── data/                               # Downloaded datasets (not in git)
│   ├── raw/
│   └── processed/
│
└── notebooks/
    └── eda.ipynb                       # Exploratory data analysis
```

---

## Configuration

Edit `configs/tb.yaml` or `configs/pneumonia.yaml` to adjust:

```yaml
training:
  epochs: 30
  batch_size: 16        # Lower to 8 if running on CPU with limited RAM
  learning_rate: 0.001

model:
  backbone: efficientnet_b0   # Upgrade to efficientnet_b2 for better accuracy
  dropout: 0.3
```

---

## Hardware Requirements

| Setup | Expected Training Time |
|---|---|
| GPU (CUDA) | ~30 min per model |
| Mac M1/M2 (MPS) | ~1–2 hours per model |
| CPU only | ~4–8 hours per model |

For inference (demo only), **no GPU is required**. Runs fine on any laptop.

---

## Deploy to HuggingFace Spaces (Free)

1. Create a free account at https://huggingface.co
2. New Space → Gradio → upload this repo
3. Add your trained `.pth` files to `models/`
4. Done — anyone can use your demo via a public URL

---

## Disclaimer

> ⚠️ This project is for **research and educational purposes only**.
> It has not been clinically validated and should not be used for medical diagnosis.
> Always consult a qualified radiologist or physician.

---

## Datasets

- **Montgomery TB**: [Kaggle — raddar/tuberculosis-chest-xrays-montgomery](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery)
- **Shenzhen TB**: [Kaggle — raddar/tuberculosis-chest-xrays-shenzhen](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen)
- **Chest X-Ray Pneumonia**: [Kaggle — paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
