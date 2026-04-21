"""Train a selectable classical segmentation baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

from cell_tracking.config import artifacts_path
from cell_tracking.features import ImageMaskPair, process_images
from cell_tracking.models import classical as classical_module
from cell_tracking.splits import default_frame_indices, format_frame_number, parse_frame_spec

console = Console()


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
    parser.add_argument(
        "--no-enhanced-features",
        action="store_true",
        help="Use raw pixel features instead of scikit-image enhanced features",
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
    
    console.print(f"[bold blue]Training {args.model.upper()} model[/bold blue]")
    console.print(f"  Dataset: {args.dataset}")
    console.print(f"  Track: {args.track}")
    console.print(f"  Frames: {frame_indices[0]}-{frame_indices[-1]} ({len(frame_indices)} total)")
    console.print(f"  Window size: {args.window}")
    console.print(f"  Samples per image: {args.samples}")
    console.print()
    
    console.print("[bold]Extracting features from images...[/bold]")
    df = process_images(
        pairs, 
        args.window, 
        args.samples, 
        args.pct_fg,
        use_enhanced_features=not args.no_enhanced_features
    )
    
    x = df.drop(columns=["label", "image"]).values
    y = df["label"].values
    
    console.print(f"[green]✓ Extracted {len(x):,} training samples with {x.shape[1]} features[/green]")
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
    ) as progress:
        progress.add_task(f"Training {args.model} model...", total=None)
        model = classical_module.train_model(x, y, args.model)
    
    classical_module.save_model(model, args.model_path)
    
    console.print()
    console.print("[bold green]✓ Training complete![/bold green]")
    console.print(f"  Model saved: {args.model_path}")
    console.print(f"  Feature type: {'scikit-image enhanced' if not args.no_enhanced_features else 'raw pixel'}")
    console.print(f"  Training samples: {len(x):,}")
    console.print(f"  Feature dimensions: {x.shape[1]}")


if __name__ == "__main__":
    main()
