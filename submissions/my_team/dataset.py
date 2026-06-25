from pathlib import Path
from PIL import Image
import torch
from labels import HF_INDEX_TO_NAME, HF_INDEX_TO_IDX, TARGET_HF_INDICES


class AugmentationDataset(torch.utils.data.Dataset):
    """
    Loads instructor-provided augmented images from augmentations/color_jitter
    or augmentations/random_rotation folders.
    """
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples = []

        for hf_idx in sorted(TARGET_HF_INDICES):
            class_name = HF_INDEX_TO_NAME[hf_idx]
            class_dir  = root / class_name
            local_idx  = HF_INDEX_TO_IDX[hf_idx]

            if not class_dir.exists():
                continue

            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, local_idx))

        print(f"Loaded {len(self.samples)} images from {root.name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label