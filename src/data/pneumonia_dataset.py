from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from src.utils.transforms import get_train_transforms, get_val_transforms
from src.utils.severity_mapper import map_label


class PneumoniaDataset(Dataset):
    def __init__(self, config, split="train"):
        self.root = Path(config["dataset_path"]) / split
        self.image_paths = list(self.root.glob("*/*.jpeg"))

        self.transform = (
            get_train_transforms(config["image_size"])
            if split == "train"
            else get_val_transforms(config["image_size"])
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")
        label_str = img_path.parent.name
        label = map_label(label_str)

        image = self.transform(image)

        return image, label
