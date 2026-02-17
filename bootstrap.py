from pathlib import Path

folders = [
    "configs",
    "data/raw",
    "data/processed",
    "src/data",
    "src/models",
    "src/training",
    "src/evaluation",
    "src/utils",
    "tasks/brain_severity",
    "tasks/pneumonia_severity",
    "notebooks",
]

files = [
    "README.md",
    "requirements.txt",
    "setup.py",
    "configs/brain_ct.yaml",
    "configs/pneumonia.yaml",
    "src/data/dataset_factory.py",
    "src/data/brain_ct_dataset.py",
    "src/data/pneumonia_dataset.py",
    "src/models/model_factory.py",
    "src/models/cnn_backbone.py",
    "src/training/train.py",
    "src/training/trainer.py",
    "src/evaluation/metrics.py",
    "src/utils/severity_mapper.py",
    "src/utils/transforms.py",
    "tasks/brain_severity/run_training.py",
    "tasks/pneumonia_severity/run_training.py",
    "notebooks/eda.ipynb",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)

for file in files:
    Path(file).touch(exist_ok=True)

print("✅ Project structure created.")
