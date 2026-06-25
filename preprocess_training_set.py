"""
Apply train.py's build_transform to every image in dataset/training_set
and save the results under dataset/processed_training_set with the same
class subfolder layout and filenames (as .jpg images).
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "submissions" / "my_team"))

from train import IMAGENET_MEAN, IMAGENET_STD, build_transform  # noqa: E402

SOURCE_ROOT = PROJECT_ROOT / "dataset" / "training_set"
TARGET_ROOT = PROJECT_ROOT / "dataset" / "processed_training_set"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)
TO_PIL = transforms.ToPILImage()


def tensor_to_jpeg_image(tensor: torch.Tensor) -> Image.Image:
    """Invert ImageNet normalization so the tensor can be saved as a JPEG."""
    image = tensor * STD + MEAN
    image = image.clamp(0.0, 1.0)
    return TO_PIL(image)


def main() -> None:
    transform = build_transform()
    saved = 0
    skipped = 0
    removed_pt = 0

    for class_dir in sorted(SOURCE_ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        out_class_dir = TARGET_ROOT / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            out_path = out_class_dir / f"{image_path.stem}.jpg"
            old_pt_path = out_class_dir / f"{image_path.stem}.pt"

            if old_pt_path.exists():
                old_pt_path.unlink()
                removed_pt += 1

            if out_path.exists():
                skipped += 1
                continue

            image = Image.open(image_path).convert("RGB")
            tensor = transform(image)
            tensor_to_jpeg_image(tensor).save(out_path, quality=95)
            saved += 1

        print(f"{class_dir.name}: saved {len(image_paths)} images")

    print(
        f"Done. Saved {saved} new JPEGs, skipped {skipped} existing, "
        f"removed {removed_pt} old .pt files."
    )


if __name__ == "__main__":
    main()
