"""Data download and preparation helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm

from .config import (
    DEFAULT_ARTIFACTS,
    EVALUATION_ZIP_URL,
    SEGMEASURE_FALLBACK_URL,
    TEST_DATASETS,
    TRAINING_DATASETS,
)


BUFFER_SIZE = 1024 * 256


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(destination, "wb") as fh, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
            disable=total == 0,
        ) as bar:
            for chunk in response.iter_content(chunk_size=BUFFER_SIZE):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))


def _extract_zip(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        archive.extractall(target)


def ensure_evaluation_tools(base_dir: Path = DEFAULT_ARTIFACTS) -> Path:
    """Download the Challenge evaluation binaries if missing."""

    tools_dir = base_dir / "tools"
    segmeasure_binary = tools_dir / "Linux" / "SEGMeasure"
    if segmeasure_binary.exists():
        return tools_dir

    zip_path = tools_dir / "EvaluationSoftware.zip"
    if not zip_path.exists():
        _download_file(EVALUATION_ZIP_URL, zip_path)
    _extract_zip(zip_path, tools_dir)
    if segmeasure_binary.exists():
        segmeasure_binary.chmod(0o755)
    return tools_dir
def ensure_segmeasure_script(base_dir: Path = DEFAULT_ARTIFACTS) -> Path:
    """Download the MySEGMeasure helper script."""

    script_path = base_dir / "tools" / "MySEGMeasure.py"
    if not script_path.exists():
        _download_file(SEGMEASURE_FALLBACK_URL, script_path)
    return script_path


def ensure_dataset(name: str, split: str = "training", base_dir: Path = DEFAULT_ARTIFACTS) -> Path:
    """Ensure that a dataset split is downloaded and extracted."""

    split = split.lower()
    if split not in {"training", "test"}:
        raise ValueError("split must be 'training' or 'test'")

    url_map = TRAINING_DATASETS if split == "training" else TEST_DATASETS
    if name not in url_map:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(url_map)}")

    dataset_root = base_dir / "datasets" / name / split
    if any(dataset_root.iterdir() for _ in dataset_root.glob("*")):
        return dataset_root

    archive_dir = base_dir / "downloads"
    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archive_dir / f"{name}-{split}.zip"
    if not zip_path.exists():
        _download_file(url_map[name], zip_path)
    _extract_zip(zip_path, dataset_root)
    return dataset_root


def ensure_all(name: str, splits: Optional[Iterable[str]] = None) -> None:
    """Download evaluation tools plus the requested dataset splits."""

    ensure_evaluation_tools()
    ensure_segmeasure_script()
    splits = splits or ("training", "test")
    for split in splits:
        ensure_dataset(name, split)
