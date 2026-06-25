#!/usr/bin/env python3
"""
Classify a single image with the trained model.

Usage:
  python classify_image.py path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEAM_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_DIR))

from predict import Model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_labels() -> dict[str, str]:
    with open(PROJECT_ROOT / "labels.json", encoding="utf-8") as f:
        return json.load(f)


def preprocess(image_path: Path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def classify(image_path: Path, model: Model, labels: dict[str, str]) -> tuple[int, str]:
    x = preprocess(image_path)
    pred_idx = int(model.predict(x).item())
    return pred_idx, labels[str(pred_idx)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one image with weights.joblib")
    parser.add_argument("image", type=Path, help="Path to a .jpg/.png image")
    parser.add_argument(
        "--weights",
        type=Path,
        default=TEAM_DIR / "weights.joblib",
        help="Path to weights file (default: weights.joblib in this folder)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = Model()
    model.load(str(args.weights))

    labels = load_labels()
    pred_idx, class_name = classify(args.image, model, labels)

    print(f"Image:     {args.image}")
    print(f"Predicted: {pred_idx} ({class_name})")


if __name__ == "__main__":
    main()
