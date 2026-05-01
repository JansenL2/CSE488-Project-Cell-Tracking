"""Evaluation helpers (IoU + SEGMeasure)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import tifffile
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import io

from .config import DEFAULT_ARTIFACTS
from .data import ensure_segmeasure_script, get_segmeasure_binary


ArrayLike = np.ndarray


@dataclass
class SegMeasureResult:
    """Structured result from a SEGMeasure run."""

    mean_jaccard_index: float | None
    stdout: str
    stderr: str
    command: list[str]
    elapsed_seconds: float | None = None


def compute_jaccard_index_for_matches(ref_image: ArrayLike, seg_mask: ArrayLike) -> Tuple[float, Dict[int, float]]:
    """Compute mean IoU between reference labels and predicted mask components."""

    labeled_mask = label(seg_mask)
    mask_props = regionprops(labeled_mask)
    jaccard_indices: Dict[int, float] = {}

    for ref_label in np.unique(ref_image):
        if ref_label == 0:
            continue
        ref_object = ref_image == ref_label
        overlaps = []
        for prop in mask_props:
            intersection = np.sum(ref_object & (labeled_mask == prop.label))
            union = np.sum(ref_object | (labeled_mask == prop.label))
            if intersection > 0.5 * np.sum(ref_object):
                overlaps.append(intersection / union)
            else:
                overlaps.append(0.0)
        if overlaps:
            jaccard_indices[int(ref_label)] = max(overlaps)

    mean_iou = float(np.mean(list(jaccard_indices.values()))) if jaccard_indices else 0.0
    return mean_iou, jaccard_indices


def _parse_mean_jaccard_index(output: str) -> float | None:
    """Extract the SEG/mean Jaccard score from command output."""

    for line in output.splitlines():
        if "Mean Jaccard Index:" in line:
            _, value = line.split("Mean Jaccard Index:", maxsplit=1)
            try:
                return float(value.strip())
            except ValueError:
                return None
        if "SEG measure:" in line:
            _, value = line.split("SEG measure:", maxsplit=1)
            try:
                return float(value.strip())
            except ValueError:
                return None
    return None


def _segmeasure_pairs(gt_dir: Path, res_dir: Path) -> list[tuple[str, Path, Path]]:
    """Return matching GT/prediction frame pairs for SEGMeasure-style evaluation."""

    pairs: list[tuple[str, Path, Path]] = []
    for gt_path in sorted(gt_dir.glob("man_seg*.tif")):
        match = re.search(r"man_seg(\d+)", gt_path.stem)
        if not match:
            continue
        frame_num = match.group(1)
        pred_path = res_dir / f"mask{frame_num}.tif"
        if pred_path.exists():
            pairs.append((frame_num, gt_path, pred_path))
    return pairs


def count_segmeasure_pairs(gt_dir: Path, res_dir: Path) -> int:
    """Count the number of frame pairs that SEGMeasure will evaluate."""

    return len(_segmeasure_pairs(gt_dir, res_dir))


def run_segmeasure(
    gt_dir: Path,
    res_dir: Path,
    verbose: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SegMeasureResult:
    """Run SEGMeasure using a CTC-style staged folder layout."""

    pairs = _segmeasure_pairs(gt_dir, res_dir)
    total = len(pairs)
    start_time = time.perf_counter()

    for index, (frame_num, _gt_path, _pred_path) in enumerate(pairs, start=1):
        if progress_callback is not None:
            progress_callback(index, total, frame_num)

    staged_root = stage_ctc_segmeasure_layout(gt_dir, res_dir)
    command = _build_segmeasure_binary_command(staged_root, track_id=gt_dir.parent.name.replace("_ST", "").replace("_GT", ""))
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        if completed.returncode != 0 or _parse_mean_jaccard_index(stdout) is None:
            command = _build_segmeasure_script_command(gt_dir, res_dir, verbose=verbose)
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
    except OSError as exc:
        command = _build_segmeasure_script_command(gt_dir, res_dir, verbose=verbose)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr or str(exc)
        except OSError as fallback_exc:
            stdout = ""
            stderr = f"{exc}; fallback failed: {fallback_exc}"

    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr and verbose:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)

    return SegMeasureResult(
        mean_jaccard_index=_parse_mean_jaccard_index(stdout),
        stdout=stdout,
        stderr=stderr,
        command=[str(part) for part in command],
        elapsed_seconds=time.perf_counter() - start_time,
    )


def stage_ctc_segmeasure_layout(gt_dir: Path, res_dir: Path) -> Path:
    """Stage the selected GT/prediction pairs into a CTC-style eval layout."""

    pairs = _segmeasure_pairs(gt_dir, res_dir)
    track_id = gt_dir.parent.name.replace("_ST", "").replace("_GT", "")
    staged_root = res_dir / "ctc_eval"
    staged_gt_dir = staged_root / f"{track_id}_GT" / "SEG"
    staged_res_dir = staged_root / f"{track_id}_RES"

    shutil.rmtree(staged_root, ignore_errors=True)
    staged_gt_dir.mkdir(parents=True, exist_ok=True)
    staged_res_dir.mkdir(parents=True, exist_ok=True)

    for _frame_num, gt_path, pred_path in pairs:
        shutil.copy2(gt_path, staged_gt_dir / gt_path.name)
        pred_image = io.imread(str(pred_path))
        if pred_image.ndim > 2:
            pred_image = pred_image[..., 0]
        unique_values = np.unique(pred_image)
        if unique_values.size <= 2:
            staged_pred = label(pred_image > 0).astype(np.uint16)
        else:
            staged_pred = pred_image.astype(np.uint16)
        tifffile.imwrite(
            staged_res_dir / pred_path.name,
            staged_pred,
            compression="lzw",
            photometric="minisblack",
        )

    return staged_root


def _build_segmeasure_script_command(gt_dir: Path, res_dir: Path, verbose: bool = False) -> list[str]:
    """Build the command for the bundled ``MySEGMeasure.py`` script."""

    seg_script = ensure_segmeasure_script(DEFAULT_ARTIFACTS)
    command = [sys.executable, str(seg_script), str(gt_dir), str(res_dir)]
    if verbose:
        command.append("-v")
    return command


def _build_segmeasure_binary_command(staged_root: Path, track_id: str) -> list[str]:
    """Build the command for the official SEGMeasure binary."""

    seg_binary = get_segmeasure_binary(DEFAULT_ARTIFACTS)
    return [str(seg_binary.resolve()), str(staged_root.resolve()), track_id, "3"]


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds into a compact human-readable string."""

    if seconds is None:
        return "n/a"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def save_evaluation_report(output_dir: Path, report: Dict[str, Any]) -> tuple[Path, Path]:
    """Persist the latest evaluation summary as text and JSON files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "latest_evaluation.txt"
    json_path = output_dir / "latest_evaluation.json"

    segmeasure = report.get("segmeasure", {})
    iou_summary = report.get("iou_summary", {})
    frames = report.get("frames", {})

    text_lines = [
        f"Evaluation Timestamp: {report.get('timestamp', 'unknown')}",
        f"Dataset: {report.get('dataset', 'unknown')}",
        f"Track: {report.get('track', 'unknown')}",
        f"Model: {report.get('model', 'unknown')}",
        f"Model Path: {report.get('model_path', 'unknown')}",
        f"Window Size: {report.get('window_size', 'unknown')}",
        f"Results Directory: {report.get('results_dir', output_dir)}",
        f"Ground Truth Directory: {report.get('ground_truth_dir', 'unknown')}",
        f"Total Evaluation Time (seconds): {report.get('total_elapsed_seconds', 'n/a')}",
        f"Total Evaluation Time: {format_duration(report.get('total_elapsed_seconds'))}",
        f"Frame Evaluation Time (seconds): {report.get('frame_evaluation_elapsed_seconds', 'n/a')}",
        f"Frame Evaluation Time: {format_duration(report.get('frame_evaluation_elapsed_seconds'))}",
        f"SEGMeasure Time (seconds): {report.get('segmeasure_elapsed_seconds', segmeasure.get('elapsed_seconds'))}",
        f"SEGMeasure Time: {format_duration(report.get('segmeasure_elapsed_seconds', segmeasure.get('elapsed_seconds')))}",
        f"Frames Requested: {frames.get('requested_count', 0)}",
        f"Frames Evaluated: {frames.get('evaluated_count', 0)}",
        f"Frames Skipped: {frames.get('skipped_count', 0)}",
        "",
        "IoU Summary:",
        f"  Mean IoU: {iou_summary.get('mean', 'n/a')}",
        f"  Std Dev: {iou_summary.get('std', 'n/a')}",
        f"  Min IoU: {iou_summary.get('min', 'n/a')}",
        f"  Max IoU: {iou_summary.get('max', 'n/a')}",
        "",
        "SEGMeasure:",
        "  Running SEGMeasure...",
        f"  Mean Jaccard Index: {segmeasure.get('mean_jaccard_index', 'n/a')}",
    ]

    if segmeasure.get("stdout"):
        text_lines.extend(
            [
                "",
                "SEGMeasure Raw Output:",
                segmeasure["stdout"].rstrip(),
            ]
        )

    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return text_path, json_path


def segmeasure_result_to_dict(result: SegMeasureResult) -> Dict[str, Any]:
    """Convert a SEGMeasure result dataclass into a JSON-friendly dictionary."""

    return asdict(result)


def label_cells(seg_mask: ArrayLike) -> ArrayLike:
    """Label connected components in a binary segmentation mask.
    
    Each cell gets a unique integer label (1, 2, 3, ...).
    Background is 0.
    
    Args:
        seg_mask: Binary segmentation mask (0 = background, >0 = cells)
    
    Returns:
        Labeled array where each cell has a unique ID
    """
    return label(seg_mask)


def colorize_segmentation(seg_mask: ArrayLike, image: ArrayLike | None = None) -> ArrayLike:
    """Convert a segmentation mask to a colored RGB image with distinct colors per cell.
    
    Args:
        seg_mask: Binary or labeled segmentation mask
        image: Optional original grayscale image to overlay (will be converted to RGB)
    
    Returns:
        RGB colored image (H, W, 3) with each cell in a different color
    """
    # Label cells if not already labeled
    if np.max(seg_mask) == 1:
        labeled = label(seg_mask)
    else:
        labeled = seg_mask
    
    # Convert to RGB with distinct colors for each label
    # The 'jet' colormap provides good color separation
    colored = label2rgb(labeled, image=image, alpha=0.3, bg_label=0, colors=None)
    
    return (colored * 255).astype(np.uint8)


def save_colored_segmentation(seg_mask: ArrayLike, output_path: Path, image: ArrayLike | None = None) -> None:
    """Save a colored segmentation visualization.
    
    Args:
        seg_mask: Binary or labeled segmentation mask
        output_path: Path to save the colored image (e.g., 'colored_mask.png')
        image: Optional original image to overlay
    """
    colored = colorize_segmentation(seg_mask, image)
    io.imsave(str(output_path), colored)
