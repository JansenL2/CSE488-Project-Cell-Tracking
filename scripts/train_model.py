"""Train a selectable classical segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_tracking.config import artifacts_path
from cell_tracking.features import ImageMaskPair, process_images
from cell_tracking.models import classical as classical_module
from cell_tracking.splits import default_frame_indices, format_frame_number, parse_frame_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classical sliding-window model")
    parser.add_argument("dataset", help="Dataset name (e.g., Fluo-N2DH-GOWT1)")
    parser.add_argument("--track", default="01", help="Track identifier (01/02)")
    parser.add_argument("--window", type=int, default=5, help="Sliding window size")
    parser.add_argument("--samples", type=int, default=500, help="Samples per image")
    parser.add_argument("--pct-fg", type=float, default=0.5, help="Foreground sampling ratio")
    parser.add_argument(
        "--model",
        choices=classical_module.MODEL_CHOICES,
        default="svm",
        help="Classical model to train",
    )
    parser.add_argument(
        "--frames",
        help="Comma-separated frame list/ranges for training, e.g. '0-19,25,30-35'",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=artifacts_path("models", "svm.pkl"),
        help="Output pickle destination",
    )
    args = parser.parse_args()

    dataset_root = artifacts_path("datasets", args.dataset, "training", args.dataset)
    frame_indices = parse_frame_spec(args.frames) or default_frame_indices()
    pairs = [
        ImageMaskPair(
            dataset_root / args.track / f"t{format_frame_number(i)}.tif",
            dataset_root / f"{args.track}_ST" / "SEG" / f"man_seg{format_frame_number(i)}.tif",
        )
        for i in frame_indices
    ]
    df = process_images(pairs, args.window, args.samples, args.pct_fg)
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].values

    model = classical_module.train_model(x, y, args.model)
    classical_module.save_model(model, args.model_path)
    print(f"Saved {args.model} model to {args.model_path}")
    print(f"Training frames: {', '.join(format_frame_number(i) for i in frame_indices)}")


if __name__ == "__main__":
    main()
