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


def format_frame_number(index: int) -> str:
    """Return a zero-padded frame identifier."""

    return f"{index:03d}"
