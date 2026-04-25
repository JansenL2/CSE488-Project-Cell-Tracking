"""Typer-based command-line interface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from typing import List

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from skimage import io

from .config import artifacts_path
from .data import ensure_all
from .evaluation import (
    count_segmeasure_pairs,
    compute_jaccard_index_for_matches,
    run_segmeasure,
    save_evaluation_report,
    segmeasure_result_to_dict,
)
from .features import ImageMaskPair, process_images
from .models import svm as svm_module

app = typer.Typer(help="Utilities for the CSE488 cell tracking project")
console = Console()


@app.command()
def setup(dataset: str, splits: List[str] = typer.Option(["training", "test"], help="Dataset splits to fetch")):
    """Download evaluation software plus dataset splits."""

    ensure_all(dataset, splits)
    typer.echo("Assets ready under artifacts/.")


@app.command()
def train_svm(
    dataset: str = typer.Argument(..., help="Dataset name"),
    track: str = typer.Option("01", help="Track identifier"),
    window: int = typer.Option(5, help="Sliding window size"),
    samples: int = typer.Option(500, help="Samples per image"),
    kernel: str = typer.Option("rbf", help="SVM kernel"),
    pct_fg: float = typer.Option(0.5, min=0.1, max=0.9, help="Foreground sampling rate"),
    model_path: Path = typer.Option(artifacts_path("models", "svm.pkl"), help="Output model path"),
):
    """Train a simple sliding-window SVM baseline."""

    dataset_root = artifacts_path("datasets", dataset, "training")
    pairs = [
        ImageMaskPair(
            dataset_root / track / f"t{i:03d}.tif",
            dataset_root / f"{track}_ST" / "SEG" / f"man_seg{i:03d}.tif",
        )
        for i in range(3)
    ]
    df = process_images(pairs, window_size=window, samples_per_image=samples, pct_fg=pct_fg)
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].values
    model = svm_module.train_svm(x, y, kernel=kernel)
    svm_module.save_model(model, model_path)
    typer.echo(f"Saved model to {model_path}")


@app.command()
def eval_svm(
    dataset: str = typer.Argument(..., help="Dataset name"),
    track: str = typer.Option("01", help="Track identifier"),
    window: int = typer.Option(5, help="Window size used during training"),
    model_path: Path = typer.Option(artifacts_path("models", "svm.pkl"), help="Trained model path"),
    verbose: bool = typer.Option(False, help="Run SEGMeasure verbosely"),
):
    """Generate predictions for a single frame and compute IoU + SEGMeasure."""

    evaluation_start = time.perf_counter()
    model = svm_module.load_model(model_path)
    dataset_root = artifacts_path("datasets", dataset, "training")
    frame_eval_start = time.perf_counter()
    image = io.imread(str(dataset_root / track / "t000.tif"))
    preds = svm_module.predict_image(image, model, window)
    mask = preds.reshape(image.shape)

    gt = io.imread(str(dataset_root / f"{track}_ST" / "SEG" / "man_seg000.tif"))
    mean_iou, per_object = compute_jaccard_index_for_matches(gt, mask > 0.5)
    frame_evaluation_elapsed_seconds = time.perf_counter() - frame_eval_start
    typer.echo(f"Mean IoU: {mean_iou:.3f}")
    for label, iou in per_object.items():
        typer.echo(f"  Label {label}: {iou:.3f}")

    gt_dir = dataset_root / f"{track}_ST" / "SEG"
    pred_dir = artifacts_path("results", dataset, track)
    pred_dir.mkdir(parents=True, exist_ok=True)
    io.imsave(str(pred_dir / "mask000.tif"), (mask > 0.5).astype(np.uint8) * 255)
    console.print("[bold blue]Running SEGMeasure...[/bold blue]")
    segmeasure_total = count_segmeasure_pairs(gt_dir, pred_dir)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Computing SEGMeasure score...", total=max(segmeasure_total, 1))
        segmeasure_result = run_segmeasure(
            gt_dir,
            pred_dir,
            verbose,
            progress_callback=lambda completed, _total, frame_num: progress.update(
                task,
                completed=completed,
                description=f"Computing SEGMeasure score... frame {frame_num}",
            ),
        )
    total_elapsed_seconds = time.perf_counter() - evaluation_start

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "track": track,
        "model": "svm",
        "model_path": str(model_path.resolve()),
        "window_size": window,
        "results_dir": str(pred_dir.resolve()),
        "ground_truth_dir": str(gt_dir.resolve()),
        "total_elapsed_seconds": total_elapsed_seconds,
        "frame_evaluation_elapsed_seconds": frame_evaluation_elapsed_seconds,
        "segmeasure_elapsed_seconds": segmeasure_result.elapsed_seconds,
        "frames": {
            "requested_count": 1,
            "evaluated_count": 1,
            "skipped_count": 0,
            "requested": ["000"],
            "evaluated": ["000"],
            "skipped": [],
        },
        "iou_summary": {
            "mean": float(mean_iou),
            "std": 0.0,
            "min": float(mean_iou),
            "max": float(mean_iou),
        },
        "frame_metrics": [
            {
                "frame": "000",
                "mean_iou": float(mean_iou),
                "matched_objects": len(per_object),
                "per_object_iou": {str(label): float(iou) for label, iou in per_object.items()},
            }
        ],
        "segmeasure": segmeasure_result_to_dict(segmeasure_result),
    }
    text_report_path, json_report_path = save_evaluation_report(pred_dir, report)
    typer.echo(f"Saved latest evaluation report to {text_report_path} and {json_report_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
