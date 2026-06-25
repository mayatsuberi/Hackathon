from pathlib import Path

import joblib
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder

from base_model import ImageNetSubset
from model import ModelArchitecture


DATA_ROOT = Path("../../dataset")
OUTPUT = Path("weights.joblib")

IMAGE_SIZE = 224
BATCH_SIZE = 32


class PadToSquare:
    """
    A custom PyTorch Transform class.
    It pads the image with a specified fill value to make it a perfect square,
    preventing distortion during the resizing process.
    """
    def __init__(self, fill=0):
        # Fill value for the padded area (0 = black)
        self.fill = fill

    def __call__(self, img):
        # img is expected to be a PIL Image
        w, h = img.size
        max_wh = max(w, h)
        
        # Calculate the required padding for each side
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        
        # Padding format: (left, top, right, bottom)
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        
        return F.pad(img, padding, self.fill, 'constant')


def main():
    """
    Full training pipeline.

    This script must create weights.joblib.
    """

    # 1. Define the transform pipeline using the custom padding class
    transform_pipeline = transforms.Compose([
        PadToSquare(fill=0),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 2. Load the training dataset
    # Navigate to the raw images directory within the dataset folder
    dataset_path = DATA_ROOT / "train_set" 
    
    train_dataset = ImageFolder(root=str(dataset_path), transform=transform_pipeline)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Loaded {len(train_dataset)} images.")

    # TODO: create your model

    # Save trained model weights to weights.joblib
    # Important: Move the model to CPU before saving to ensure compatibility 
    # during the automated evaluation process.
    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, "weights.joblib")
    print("Saved trained weights.joblib")


if __name__ == "__main__":
    main()