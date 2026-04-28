# Cell Segmentation Showcase

This page summarizes the undergraduate classical segmentation portion of the project. The experiments use `Fluo-N2DH-GOWT1`, track `01`, and compare three trainable classical methods under the same feature extraction and evaluation pipeline.

## Overview

- Dataset: `Fluo-N2DH-GOWT1`
- Track: `01`
- Methods compared: `svm`, `logreg`, `rf`
- Feature style: sliding-window pixel neighborhoods with enhanced scikit-image-derived features
- Evaluation: held-out silver-truth frames with IoU and SEG / Mean Jaccard

## Split Strategy

I created a train/validation/test split from the labeled silver-truth frames in the challenge `training` data:

- Training: `0-54`
- Validation: `55-72`
- Test: `73-91`

This is the evaluation protocol used for the reported results below.

## Quantitative Results

Held-out evaluation on frames `73-91`:

| Method | Mean IoU | SEG / Mean Jaccard | Notes |
| ------ | -------- | ------------------ | ----- |
| SVM | `0.8896` | `0.8895` | Strong baseline, but slower evaluation and slightly lower final score |
| Logistic Regression | `0.8928` | `0.8926` | Competitive and fast, but still behind RF |
| Random Forest | `0.9037` | `0.9035` | Best classical result in this repo |

## What Worked

- Using the same split and window size across all methods made the comparison straightforward.
- The enhanced sliding-window features were strong enough for all three classical models to produce useful segmentations.
- Random Forest handled the local texture and boundary cues best among the classical baselines tested here.

## What Still Needs Improvement

- Some difficult frames still show weaker boundary separation and merged cells.
- The current workflow is semantic foreground/background segmentation rather than full instance-aware segmentation.
- More hyperparameter tuning on the validation split would likely improve the weaker models.

## Reproduction

```bash
conda env create -f environment.yml
conda activate cse488-cell-tracking
python scripts/setup_data.py Fluo-N2DH-GOWT1 --splits training test

python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model svm --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/svm.pkl
python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model logreg --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/logreg.pkl
python scripts/train_model.py Fluo-N2DH-GOWT1 --track 01 --model rf --frames 0-54 --window 5 --samples 500 --model-path artifacts/models/rf.pkl

python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model svm --frames 73-91 --window 5 --model-path artifacts/models/svm.pkl
python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model logreg --frames 73-91 --window 5 --model-path artifacts/models/logreg.pkl
python scripts/eval_seg.py Fluo-N2DH-GOWT1 --track 01 --model rf --frames 73-91 --window 5 --model-path artifacts/models/rf.pkl
```

## Deliverable Note

If you publish this page, replace this short summary with your preferred final wording and add qualitative images or videos from `docs/assets/` if you want a stronger public-facing presentation.
