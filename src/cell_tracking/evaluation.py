"""Evaluation helpers (IoU + SEGMeasure)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage import io

from .config import DEFAULT_ARTIFACTS
from .data import ensure_evaluation_tools, ensure_segmeasure_script


ArrayLike = np.ndarray


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


def run_segmeasure(gt_dir: Path, res_dir: Path, verbose: bool = False) -> None:
    """Invoke the official SEGMeasure script on GT vs result folders."""

    base_dir = DEFAULT_ARTIFACTS
    tools_dir = ensure_evaluation_tools(base_dir)
    seg_script = ensure_segmeasure_script(base_dir)
    env = dict(PATH=f"{tools_dir}/Linux:" + subprocess.os.environ.get("PATH", ""))
    cmd = [
        "python",
        str(seg_script),
        str(gt_dir),
        str(res_dir),
    ]
    if verbose:
        cmd.append("-v")
    subprocess.run(cmd, check=True, env={**subprocess.os.environ, **env})


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
