"""Evaluate predictions with IoU + SEGMeasure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage import io

from cell_tracking.config import artifacts_path
from cell_tracking.evaluation import compute_jaccard_index_for_matches, run_segmeasure
from cell_tracking.models import svm as svm_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SVM predictions")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--track", default="01", help="Track identifier")
    parser.add_argument("--window", type=int, default=5, help="Window size used for training")
    parser.add_argument("--model-path", type=Path, default=artifacts_path("models", "svm.pkl"))
    parser.add_argument("--verbose", action="store_true", help="Print SEGMeasure output")
    args = parser.parse_args()

    model = svm_module.load_model(args.model_path)
    dataset_root = artifacts_path("datasets", args.dataset, "training", args.dataset)
    image = io.imread(str(dataset_root / args.track / "t000.tif"))
    gt = io.imread(str(dataset_root / f"{args.track}_ST" / "SEG" / "man_seg000.tif"))

    preds = svm_module.predict_image(image, model, args.window)
    mask = (preds.reshape(image.shape) > 0.5).astype(np.uint8)
    mean_iou, per_object = compute_jaccard_index_for_matches(gt, mask)
    print(f"Mean IoU: {mean_iou:.3f}")
    for label, iou in per_object.items():
        print(f"  Label {label}: {iou:.3f}")

    pred_dir = artifacts_path("results", args.dataset, args.track)
    pred_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(str(pred_dir / "mask000.tif"), mask * 255)
    run_segmeasure(
        dataset_root / f"{args.track}_ST" / "SEG",
        pred_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
