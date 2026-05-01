"""Typer-based command-line interface for the undergraduate workflow."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import tifffile
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from skimage import io

from .config import artifacts_path
from .data import ensure_all
from .evaluation import (
    count_segmeasure_pairs,
    compute_jaccard_index_for_matches,
    label_cells,
    run_segmeasure,
    save_colored_segmentation,
    save_evaluation_report,
    segmeasure_result_to_dict,
)
from .features import ImageMaskPair, process_images
from .models import classical as classical_module
from .splits import default_frame_indices, format_frame_number, parse_frame_spec, train_validation_split

app = typer.Typer(help="Utilities for the CSE488 undergraduate cell segmentation project")
console = Console()


@app.command()
def setup(
    dataset: str = typer.Argument(..., help="Dataset name"),
    splits: list[str] = typer.Option(["training", "test"], help="Dataset splits to fetch"),
) -> None:
    """Download evaluation software plus dataset splits."""

    ensure_all(dataset, splits)
    console.print("[green]Assets ready under artifacts/[/green]")


@app.command()
def split_demo(
    total_frames: int = typer.Option(92, help="Total number of frames"),
    train_fraction: float = typer.Option(0.6, help="Fraction for training"),
    val_fraction: float = typer.Option(0.2, help="Fraction for validation"),
) -> None:
    """Show a recommended train/validation/test frame split."""

    train_frames, val_frames, test_frames = train_validation_split(
        total_frames=total_frames,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    console.print("[bold blue]Recommended Frame Split[/bold blue]")
    console.print(f"  Training frames ({len(train_frames)}): {train_frames[0]}-{train_frames[-1]}")
    console.print(f"  Validation frames ({len(val_frames)}): {val_frames[0]}-{val_frames[-1]}")
    console.print(f"  Test frames ({len(test_frames)}): {test_frames[0]}-{test_frames[-1]}")
    console.print()
    console.print("[bold]Frame specs for scripts/CLI[/bold]")
    console.print(f"  Training:   --frames {train_frames[0]}-{train_frames[-1]}")
    console.print(f"  Validation: --frames {val_frames[0]}-{val_frames[-1]}")
    console.print(f"  Testing:    --frames {test_frames[0]}-{test_frames[-1]}")


@app.command()
def train(
    dataset: str = typer.Argument(..., help="Dataset name"),
    track: str = typer.Option("01", help="Track identifier"),
    window: int = typer.Option(5, help="Sliding window size"),
    samples: int = typer.Option(500, help="Samples per image"),
    pct_fg: float = typer.Option(0.5, help="Foreground sampling ratio"),
    model: str = typer.Option("svm", help="Classical model to train: svm, logreg, rf"),
    frames: str | None = typer.Option(None, help="Comma-separated frame list/ranges, e.g. 0-54"),
    model_path: Path = typer.Option(artifacts_path("models", "svm.pkl"), help="Output model path"),
    no_enhanced_features: bool = typer.Option(False, help="Use raw pixel windows instead of enhanced features"),
) -> None:
    """Train one of the undergraduate classical baselines."""

    dataset_root = artifacts_path("datasets", dataset, "training", dataset)
    frame_indices = parse_frame_spec(frames) or default_frame_indices()
    pairs = [
        ImageMaskPair(
            dataset_root / track / f"t{format_frame_number(i)}.tif",
            dataset_root / f"{track}_ST" / "SEG" / f"man_seg{format_frame_number(i)}.tif",
        )
        for i in frame_indices
    ]

    console.print(f"[bold blue]Training {model.upper()} model[/bold blue]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Track: {track}")
    console.print(f"  Frames: {frame_indices[0]}-{frame_indices[-1]} ({len(frame_indices)} total)")
    console.print(f"  Window size: {window}")
    console.print(f"  Samples per image: {samples}")
    console.print()

    df = process_images(
        pairs,
        window,
        samples,
        pct_fg,
        use_enhanced_features=not no_enhanced_features,
    )
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].values

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:
        progress.add_task(f"Training {model} model...", total=None)
        trained_model = classical_module.train_model(x, y, model)

    classical_module.save_model(trained_model, model_path)
    console.print("[bold green]Training complete[/bold green]")
    console.print(f"  Model saved: {model_path}")
    console.print(f"  Training samples: {len(x):,}")
    console.print(f"  Feature dimensions: {x.shape[1]}")


@app.command()
def evaluate(
    dataset: str = typer.Argument(..., help="Dataset name"),
    track: str = typer.Option("01", help="Track identifier"),
    window: int = typer.Option(5, help="Window size used for training"),
    model_path: Path = typer.Option(artifacts_path("models", "svm.pkl"), help="Trained model path"),
    model: str = typer.Option("svm", help="Model family used for output naming"),
    frames: str | None = typer.Option(None, help="Comma-separated frame list/ranges to evaluate"),
    verbose: bool = typer.Option(False, help="Print SEGMeasure output"),
) -> None:
    """Evaluate a trained classical model on held-out frames."""

    evaluation_start = time.perf_counter()
    trained_model = classical_module.load_model(model_path)
    dataset_root = artifacts_path("datasets", dataset, "training", dataset)

    track_dir = dataset_root / track
    frame_files = sorted(track_dir.glob("t*.tif"))
    selected_frames = parse_frame_spec(frames)
    if selected_frames is not None:
        selected_names = {f"t{frame:03d}" for frame in selected_frames}
        frame_files = [frame_path for frame_path in frame_files if frame_path.stem in selected_names]

    if not frame_files:
        raise typer.BadParameter(f"No frames found for track {track} in {track_dir}")

    all_ious: list[float] = []
    evaluated_frames: list[dict[str, object]] = []
    skipped_frames: list[str] = []
    pred_dir = artifacts_path("results", dataset, track, model)
    pred_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold blue]Evaluating {model.upper()} model[/bold blue]")
    console.print(f"  Dataset: {dataset}")
    console.print(f"  Track: {track}")
    console.print(f"  Frames to evaluate: {len(frame_files)}")
    console.print(f"  Results dir: {pred_dir}")
    console.print()

    frame_eval_start = time.perf_counter()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Evaluating frames...", total=len(frame_files))

        for frame_path in frame_files:
            frame_num = frame_path.stem[1:]
            image = io.imread(str(frame_path))
            gt_path = dataset_root / f"{track}_ST" / "SEG" / f"man_seg{frame_num}.tif"

            if not gt_path.exists():
                skipped_frames.append(frame_num)
                progress.update(task, advance=1)
                continue

            gt = io.imread(str(gt_path))
            use_probabilities = model in {"svm", "logreg", "rf"}
            preds = classical_module.predict_image(
                image,
                trained_model,
                window,
                return_probabilities=use_probabilities,
            )
            mask = (preds.reshape(image.shape) > 0.5).astype(np.uint8)

            mean_iou, per_object = compute_jaccard_index_for_matches(gt, mask)
            all_ious.append(mean_iou)
            evaluated_frames.append(
                {
                    "frame": frame_num,
                    "mean_iou": float(mean_iou),
                    "matched_objects": len(per_object),
                    "per_object_iou": {str(label): float(iou) for label, iou in per_object.items()},
                }
            )

            labeled_mask = label_cells(mask).astype(np.uint16)
            tifffile.imwrite(
                pred_dir / f"mask{frame_num}.tif",
                labeled_mask,
                compression="lzw",
                photometric="minisblack",
            )
            save_colored_segmentation(labeled_mask, pred_dir / f"colored_mask{frame_num}.png", image=image)
            progress.update(task, advance=1)

    frame_evaluation_elapsed_seconds = time.perf_counter() - frame_eval_start

    if all_ious:
        console.print("[bold green]Evaluation Summary[/bold green]")
        console.print(f"  Overall Mean IoU: [cyan]{np.mean(all_ious):.4f}[/cyan]")
        console.print(f"  Std Dev: {np.std(all_ious):.4f}")
        console.print(f"  Min IoU: {np.min(all_ious):.4f}")
        console.print(f"  Max IoU: {np.max(all_ious):.4f}")

    console.print()
    console.print("[bold blue]Running SEGMeasure...[/bold blue]")
    segmeasure_total = count_segmeasure_pairs(dataset_root / f"{track}_ST" / "SEG", pred_dir)
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
            dataset_root / f"{track}_ST" / "SEG",
            pred_dir,
            verbose=verbose,
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
        "model": model,
        "model_path": str(model_path.resolve()),
        "window_size": window,
        "results_dir": str(pred_dir.resolve()),
        "ground_truth_dir": str((dataset_root / f"{track}_ST" / "SEG").resolve()),
        "total_elapsed_seconds": total_elapsed_seconds,
        "frame_evaluation_elapsed_seconds": frame_evaluation_elapsed_seconds,
        "segmeasure_elapsed_seconds": segmeasure_result.elapsed_seconds,
        "frames": {
            "requested_count": len(frame_files),
            "evaluated_count": len(evaluated_frames),
            "skipped_count": len(skipped_frames),
            "requested": [frame_path.stem[1:] for frame_path in frame_files],
            "evaluated": [frame["frame"] for frame in evaluated_frames],
            "skipped": skipped_frames,
        },
        "iou_summary": {
            "mean": float(np.mean(all_ious)) if all_ious else None,
            "std": float(np.std(all_ious)) if all_ious else None,
            "min": float(np.min(all_ious)) if all_ious else None,
            "max": float(np.max(all_ious)) if all_ious else None,
        },
        "frame_metrics": evaluated_frames,
        "segmeasure": segmeasure_result_to_dict(segmeasure_result),
    }
    text_report_path, json_report_path = save_evaluation_report(pred_dir, report)
    console.print("[bold green]Saved latest evaluation report[/bold green]")
    console.print(f"  Text: {text_report_path}")
    console.print(f"  JSON: {json_report_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
