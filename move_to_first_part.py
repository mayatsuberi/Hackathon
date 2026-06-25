"""
Move a random subset of images from dataset/data to dataset/first_part.

The same number of images is selected from each subfolder, and each image
is moved into the matching subfolder under first_part.
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def move_images_per_subfolder(
    source_root: Path,
    target_root: Path,
    count_per_subfolder: int,
    seed: int | None = None,
) -> int:
    if count_per_subfolder <= 0:
        raise ValueError("count_per_subfolder must be a positive integer")

    subfolders = sorted(path for path in source_root.iterdir() if path.is_dir())
    if not subfolders:
        raise FileNotFoundError(f"No subfolders found in {source_root}")

    rng = random.Random(seed)
    moved_total = 0

    for subfolder in subfolders:
        images = list_images(subfolder)
        if len(images) < count_per_subfolder:
            raise ValueError(
                f"Not enough images in {subfolder.name}: "
                f"requested {count_per_subfolder}, found {len(images)}"
            )

        selected = rng.sample(images, count_per_subfolder)
        destination_dir = target_root / subfolder.name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for image_path in selected:
            shutil.move(str(image_path), str(destination_dir / image_path.name))

        moved_total += len(selected)
        print(f"{subfolder.name}: moved {len(selected)} images")

    return moved_total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly move the same number of images from each subfolder "
            "in dataset/data to the matching subfolder in dataset/first_part."
        )
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of images to move from each subfolder",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Source dataset root (default: dataset/data)",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).resolve().parent / "first_part",
        help="Target dataset root (default: dataset/first_part)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.is_dir():
        raise FileNotFoundError(f"Source folder not found: {args.source}")

    args.target.mkdir(parents=True, exist_ok=True)

    moved_total = move_images_per_subfolder(
        source_root=args.source,
        target_root=args.target,
        count_per_subfolder=args.count,
        seed=args.seed,
    )

    subfolder_count = len([path for path in args.source.iterdir() if path.is_dir()])
    print(
        f"\nDone. Moved {moved_total} images "
        f"({args.count} per subfolder across {subfolder_count} subfolders)."
    )


if __name__ == "__main__":
    main()
