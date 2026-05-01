"""Evaluate classical model predictions with IoU + SEGMeasure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import tifffile
from skimage import io
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from cell_tracking.config import artifacts_path
from cell_tracking.evaluation import (
    count_segmeasure_pairs,
    compute_jaccard_index_for_matches,
    label_cells,
    run_segmeasure,
    save_colored_segmentation,
    save_evaluation_report,
    segmeasure_result_to_dict,
)
from cell_tracking.models import classical as classical_module
from cell_tracking.splits import parse_frame_spec

console = Console()


def main() -> None:
    evaluation_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Evaluate classical model predictions")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("--track", default="01", help="Track identifier")
    parser.add_argument("--window", type=int, default=5, help="Window size used for training")
    parser.add_argument("--model-path", type=Path, default=artifacts_path("models", "svm.pkl"))
    parser.add_argument(
        "--model",
        choices=classical_module.MODEL_CHOICES,
        default="svm",
        help="Model family used for naming outputs and probability handling",
    )
    parser.add_argument(
        "--frames",
        help="Comma-separated frame list/ranges to evaluate, e.g. '20-29'",
    )
    parser.add_argument("--verbose", action="store_true", help="Print SEGMeasure output")
    args = parser.parse_args()

    model = classical_module.load_model(args.model_path)
    dataset_root = artifacts_path("datasets", args.dataset, "training", args.dataset)
    
    # Find all frames
    track_dir = dataset_root / args.track
    frame_files = sorted(track_dir.glob("t*.tif"))
    selected_frames = parse_frame_spec(args.frames)
    if selected_frames is not None:
        selected_names = {f"t{frame:03d}" for frame in selected_frames}
        frame_files = [frame_path for frame_path in frame_files if frame_path.stem in selected_names]
    
    if not frame_files:
        console.print(f"[red]Error:[/red] No frames found in {track_dir}")
        return
    
    # Process all frames
    all_ious = []
    evaluated_frames = []
    skipped_frames = []
    pred_dir = artifacts_path("results", args.dataset, args.track, args.model)
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Evaluating {args.model.upper()} model[/bold blue]")
    console.print(f"  Dataset: {args.dataset}")
    console.print(f"  Track: {args.track}")
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
            frame_num = frame_path.stem[1:]  # Extract "000" from "t000.tif"
            
            image = io.imread(str(frame_path))
            gt_path = dataset_root / f"{args.track}_ST" / "SEG" / f"man_seg{frame_num}.tif"
            
            if not gt_path.exists():
                console.print(f"[yellow]Skipping {frame_num}: no ground truth[/yellow]")
                skipped_frames.append(frame_num)
                progress.update(task, advance=1)
                continue
            
            gt = io.imread(str(gt_path))
            
            # Predict
            use_probabilities = args.model in {"svm", "logreg", "rf"}
            preds = classical_module.predict_image(
                image,
                model,
                args.window,
                return_probabilities=use_probabilities,
            )
            mask = (preds.reshape(image.shape) > 0.5).astype(np.uint8)
            
            # Compute IoU
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
            
            # Save labeled instance mask for Cell Tracking Challenge evaluation
            labeled_mask = label_cells(mask).astype(np.uint16)
            tifffile.imwrite(
                pred_dir / f"mask{frame_num}.tif",
                labeled_mask,
                compression="lzw",
                photometric="minisblack",
            )
            
            # Save colored segmentation (with each cell in a different color)
            save_colored_segmentation(
                labeled_mask,
                pred_dir / f"colored_mask{frame_num}.png",
                image=image
            )
            
            progress.update(task, advance=1)
    frame_evaluation_elapsed_seconds = time.perf_counter() - frame_eval_start
    
    # Summary with rich formatting
    console.print()
    if all_ious:
        console.print("[bold green]Evaluation Summary[/bold green]")
        console.print(f"  Overall Mean IoU: [cyan]{np.mean(all_ious):.4f}[/cyan]")
        console.print(f"  Std Dev: {np.std(all_ious):.4f}")
        console.print(f"  Min IoU: {np.min(all_ious):.4f}")
        console.print(f"  Max IoU: {np.max(all_ious):.4f}")
        console.print()
        console.print(f"[bold]Outputs saved to: {pred_dir}[/bold]")
        console.print("  - [cyan]mask*.tif[/cyan] - Labeled instance masks for CTC evaluation")
        console.print("  - [green]colored_mask*.png[/green] - Colored cell visualization (each cell has a unique color)")
    
    # Run SEGMeasure on all predictions
    console.print()
    console.print("[bold blue]Running SEGMeasure...[/bold blue]")
    segmeasure_total = count_segmeasure_pairs(dataset_root / f"{args.track}_ST" / "SEG", pred_dir)
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
            dataset_root / f"{args.track}_ST" / "SEG",
            pred_dir,
            verbose=args.verbose,
            progress_callback=lambda completed, _total, frame_num: progress.update(
                task,
                completed=completed,
                description=f"Computing SEGMeasure score... frame {frame_num}",
            ),
        )
    total_elapsed_seconds = time.perf_counter() - evaluation_start
    
    iou_summary = {
        "mean": float(np.mean(all_ious)) if all_ious else None,
        "std": float(np.std(all_ious)) if all_ious else None,
        "min": float(np.min(all_ious)) if all_ious else None,
        "max": float(np.max(all_ious)) if all_ious else None,
    }
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "track": args.track,
        "model": args.model,
        "model_path": str(args.model_path.resolve()),
        "window_size": args.window,
        "results_dir": str(pred_dir.resolve()),
        "ground_truth_dir": str((dataset_root / f"{args.track}_ST" / "SEG").resolve()),
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
        "iou_summary": iou_summary,
        "frame_metrics": evaluated_frames,
        "segmeasure": segmeasure_result_to_dict(segmeasure_result),
    }
    text_report_path, json_report_path = save_evaluation_report(pred_dir, report)

    console.print()
    console.print("[bold green]Saved latest evaluation report[/bold green]")
    console.print(f"  Text: {text_report_path}")
    console.print(f"  JSON: {json_report_path}")


if __name__ == "__main__":
    main()
