"""Helpers for selecting frame indices for train/validation splits."""

from __future__ import annotations

from typing import List


def parse_frame_spec(frame_spec: str | None) -> List[int] | None:
    """Parse a comma-separated frame specification like ``0,1,2`` or ``0-9,12``."""

    if not frame_spec:
        return None

    frames: set[int] = set()
    for part in frame_spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid frame range '{token}': end must be >= start")
            frames.update(range(start, end + 1))
        else:
            frames.add(int(token))
    return sorted(frames)


def default_frame_indices(limit: int = 3) -> List[int]:
    """Return the starter baseline frame indices used before split configuration."""

    return list(range(limit))


def train_validation_split(total_frames: int = 92, train_fraction: float = 0.6, val_fraction: float = 0.2) -> tuple[List[int], List[int], List[int]]:
    """Split frames into train/validation/test sets.
    
    Args:
        total_frames: Total number of frames available
        train_fraction: Fraction of frames for training (default 0.6 = 60%)
        val_fraction: Fraction of frames for validation (default 0.2 = 20%)
        
    Returns:
        Tuple of (train_frames, val_frames, test_frames)
        Remaining frames go to test set (default 0.2 = 20%)
    """
    test_fraction = 1.0 - train_fraction - val_fraction
    if test_fraction < 0:
        raise ValueError("train_fraction + val_fraction must be <= 1.0")
    
    train_count = int(total_frames * train_fraction)
    val_count = int(total_frames * val_fraction)
    
    train_frames = list(range(0, train_count))
    val_frames = list(range(train_count, train_count + val_count))
    test_frames = list(range(train_count + val_count, total_frames))
    
    return train_frames, val_frames, test_frames


def format_frame_number(index: int) -> str:
    """Return a zero-padded frame identifier."""

    return f"{index:03d}"
