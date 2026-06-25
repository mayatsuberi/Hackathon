import sys
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from base_model import ImageNetSubset
from model import ModelArchitecture

DATA_ROOT = PROJECT_ROOT / "dataset"
OUTPUT = Path(__file__).resolve().parent / "weights.joblib"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_DIM = 60

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def main() -> None:
    transform = build_transform()
    train_dataset = ImageNetSubset(DATA_ROOT, split="data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = ModelArchitecture(num_classes=20, hidden_dim=HIDDEN_DIM, image_size=IMAGE_SIZE)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"loss={running_loss / total:.4f} | "
            f"acc={correct / total:.4f}"
        )

    state_dict = model.cpu().state_dict()
    joblib.dump(state_dict, OUTPUT)
    print(f"Saved trained weights to {OUTPUT}")


if __name__ == "__main__":
    main()
