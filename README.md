# CSE488 Cell Segmentation Project

This repository contains my undergraduate CSE488 project for classical cell segmentation on the Cell Tracking Challenge silver-truth data. The work is implemented as reproducible Python scripts rather than notebook-only analysis.

The current experiments use the `Fluo-N2DH-GOWT1` dataset, track `01`, and compare three classical trainable segmentation methods:

- Support Vector Machine (`svm`)
- Logistic Regression (`logreg`)
- Random Forest (`rf`)

## Project Summary

I use the labeled silver-truth `training` split from the Cell Tracking Challenge and create my own train/validation/test partition from those labeled frames:

- Training: frames `0-54`
- Validation: frames `55-72`
- Test: frames `73-91`

This follows the assignment requirement to hold out part of the silver-truth data for evaluation rather than assuming the challenge-provided `test` download is labeled.

All three models are trained with the same:

- dataset and track
- frame split
- sliding-window size
- per-image sampling budget
- evaluation protocol

Predictions are evaluated with IoU summaries plus the official Cell Tracking Challenge `SEGMeasure` workflow. The evaluation code stages predictions into the expected CTC folder layout and falls back to `MySEGMeasure.py` only if the binary is unavailable.

## Current Results

Held-out evaluation on frames `73-91`:

| Model | Mean IoU | SEG / Mean Jaccard |
| ----- | -------- | ------------------ |
| `svm` | `0.8896` | `0.8895` |
| `logreg` | `0.8928` | `0.8926` |
| `rf` | `0.9037` | `0.9035` |

At the moment, the Random Forest baseline is the strongest of the three classical methods in this repo.

## Repository Layout

```text
CSE488-Project-Cell-Tracking/
├── README.md
├── pyproject.toml
├── environment.yml
├── scripts/
│   ├── setup_data.py
│   ├── demonstrate_split.py
│   ├── train_model.py
│   ├── train_svm.py
│   └── eval_seg.py
├── src/cell_tracking/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── splits.py
│   └── models/
│       └── classical.py
├── docs/
│   └── index.md
└── tests/
    └── test_features.py
```

## Setup

```bash
conda env create -f environment.yml
conda activate cse488-cell-tracking
```

If you prefer, you can also install directly with pip in an existing Python 3.11 environment:

```bash
pip install -e .[dev]
```

This also installs the `cell-tracking` CLI, which mirrors the same undergraduate workflow as the scripts.

## Quickstart

1. Download the dataset and evaluation tools:

```bash
python scripts/setup_data.py Fluo-N2DH-GOWT1 --splits training test
```

2. View the split helper output:

```bash
python scripts/demonstrate_split.py
```

3. Train the three classical baselines:

```bash
python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model svm --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/svm.pkl

python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model logreg --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/logreg.pkl

python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model rf --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/rf.pkl
```

4. Evaluate each model on the held-out test frames with the official Cell Tracking Challenge `SEGMeasure` binary:

```bash
python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model svm --frames 73-91 --window 5 --model-path artifacts/models/svm.pkl

python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model logreg --frames 73-91 --window 5 --model-path artifacts/models/logreg.pkl

python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model rf --frames 73-91 --window 5 --model-path artifacts/models/rf.pkl
```

## Reproducibility Notes

- Training uses images and silver-truth masks from the challenge `training` download.
- Local evaluation also uses the `training` download because that is where the labels live.
- The official challenge `test` download is not used for local scoring because it does not provide the same ground-truth segmentation masks for this workflow.
- To avoid leakage, always pass explicit `--frames` values during training and evaluation.

The model and evaluation scripts currently have permissive defaults intended for quick experimentation, so the commands above are the recommended reproducible runs for the project report.

## Files Produced

- Trained models: `artifacts/models/*.pkl`
- Predicted masks and reports: `artifacts/results/<dataset>/<track>/<model>/`
  The saved `mask*.tif` files are labeled instance masks formatted for CTC evaluation.
- CTC-style staged evaluation folders: `artifacts/results/<dataset>/<track>/<model>/ctc_eval/`
- Latest evaluation summaries: `latest_evaluation.txt` and `latest_evaluation.json` inside each model result directory

## Testing

Run the lightweight test suite with:

```bash
python3 -m pytest -q
```

## Report Checklist

The final PDF report should document:

- the problem and dataset
- the train/validation/test split and why it was chosen
- the three classical methods compared
- preprocessing and feature extraction
- training settings and evaluation protocol
- quantitative results and qualitative examples
- strengths, weaknesses, and next steps

## Notes

- `notebooks/00_reference_colab.ipynb` is legacy reference material only.
- The actual submission work for this project lives in the scripts and source code under `scripts/` and `src/`.
