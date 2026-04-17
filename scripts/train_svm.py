"""Train a simple SVM baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_tracking.config import artifacts_path
from cell_tracking.features import ImageMaskPair, process_images
from cell_tracking.models import svm as svm_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Train sliding-window SVM")
    parser.add_argument("dataset", help="Dataset name (e.g., Fluo-N2DH-GOWT1)")
    parser.add_argument("--track", default="01", help="Track identifier (01/02)")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size")
    parser.add_argument("--samples", type=int, default=500, help="Samples per image")
    parser.add_argument("--pct-fg", type=float, default=0.5, help="Foreground sampling ratio")
    parser.add_argument("--kernel", default="rbf", help="SVM kernel")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=artifacts_path("models", "svm.pkl"),
        help="Output pickle destination",
    )
    args = parser.parse_args()

    dataset_root = artifacts_path("datasets", args.dataset, "training", args.dataset)
    pairs = [
        ImageMaskPair(
            dataset_root / args.track / f"t{i:03d}.tif",
            dataset_root / f"{args.track}_ST" / "SEG" / f"man_seg{i:03d}.tif",
        )
        for i in range(3)
    ]
    df = process_images(pairs, args.window, args.samples, args.pct_fg)
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].values

    model = svm_module.train_svm(x, y, kernel=args.kernel)
    svm_module.save_model(model, args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
