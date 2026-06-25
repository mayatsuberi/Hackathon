#!/usr/bin/env python3
"""
Evaluate the model on a random sample of images from each class in dataset/data.

Usage:
  python evaluate_sample.py
  python evaluate_sample.py --count 20 --seed 42
  python evaluate_sample.py --output results.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TEAM_DIR) not in sys.path:
    sys.path.insert(0, str(TEAM_DIR))

from classify_image import classify, load_labels
from predict import Model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def evaluate_class(
    class_name: str,
    expected_idx: int,
    images: list[Path],
    model: Model,
    labels: dict[str, str],
) -> dict:
    successes: list[dict] = []
    failures: list[dict] = []

    for image_path in images:
        pred_idx, pred_name = classify(image_path, model, labels)
        entry = {
            "image": str(image_path.relative_to(PROJECT_ROOT)),
            "predicted_idx": pred_idx,
            "predicted_name": pred_name,
        }
        if pred_idx == expected_idx:
            successes.append(entry)
        else:
            failures.append(entry)

    total = len(images)
    failed = len(failures)
    return {
        "class": class_name,
        "expected_idx": expected_idx,
        "total": total,
        "correct": total - failed,
        "failed": failed,
        "accuracy": (total - failed) / total if total else 0.0,
        "successes": successes,
        "failures": failures,
    }


def print_report(results: list[dict]) -> None:
    total_images = sum(r["total"] for r in results)
    total_correct = sum(r["correct"] for r in results)
    total_failed = sum(r["failed"] for r in results)

    print("=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    print(f"Overall: {total_correct}/{total_images} correct ({100 * total_correct / total_images:.1f}%)")
    print(f"Failures: {total_failed}/{total_images}")
    print()

    for result in results:
        acc = 100 * result["accuracy"]
        print(f"{result['class']} (idx {result['expected_idx']}): "
              f"{result['correct']}/{result['total']} correct ({acc:.0f}%) — "
              f"{result['failed']} failed")

    print()
    print("=" * 72)
    print("PER-CLASS DETAILS")
    print("=" * 72)

    for result in results:
        print()
        print(f"--- {result['class']} ---")
        if result["successes"]:
            print("Succeeded:")
            for entry in result["successes"]:
                print(f"  OK  {entry['image']}")
        else:
            print("Succeeded: (none)")

        if result["failures"]:
            print("Failed:")
            for entry in result["failures"]:
                print(
                    f"  FAIL {entry['image']} "
                    f"-> predicted {entry['predicted_idx']} ({entry['predicted_name']})"
                )
        else:
            print("Failed: (none)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on random samples from each class in dataset/data"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Dataset root with one subfolder per class (default: dataset/data)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of random images per class (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=TEAM_DIR / "weights.joblib",
        help="Path to weights file (default: weights.joblib in this folder)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save full results as JSON",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {args.data_root}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    if args.count <= 0:
        raise ValueError("--count must be a positive integer")

    labels = load_labels()
    name_to_idx = {name: int(idx) for idx, name in labels.items()}

    subfolders = sorted(path for path in args.data_root.iterdir() if path.is_dir())
    if not subfolders:
        raise FileNotFoundError(f"No class subfolders found in {args.data_root}")

    rng = random.Random(args.seed)
    model = Model()
    model.load(str(args.weights))

    results: list[dict] = []
    for subfolder in subfolders:
        class_name = subfolder.name
        if class_name not in name_to_idx:
            raise KeyError(f"Class folder '{class_name}' not found in labels.json")

        images = list_images(subfolder)
        if len(images) < args.count:
            raise ValueError(
                f"Class '{class_name}' has only {len(images)} images, "
                f"but --count is {args.count}"
            )

        sampled = rng.sample(images, args.count)
        results.append(
            evaluate_class(class_name, name_to_idx[class_name], sampled, model, labels)
        )

    print_report(results)

    if args.output:
        payload = {
            "data_root": str(args.data_root.relative_to(PROJECT_ROOT)),
            "count_per_class": args.count,
            "seed": args.seed,
            "overall": {
                "total": sum(r["total"] for r in results),
                "correct": sum(r["correct"] for r in results),
                "failed": sum(r["failed"] for r in results),
            },
            "classes": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print()
        print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
