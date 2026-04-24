# Classical Model Comparison Report

## Project Summary

This report compares three classical machine learning models for cell segmentation on the `Fluo-N2DH-GOWT1` dataset, track `01`. All three models were trained and evaluated in the same pipeline so that the comparison is as fair as possible.

The models compared are:

- Support Vector Machine (SVM)
- Logistic Regression (LogReg)
- Random Forest (RF)

The evaluation used held-out test frames `73-91`, which corresponds to `19` total frames.

## Experimental Setup

Common training settings:

- Dataset: `Fluo-N2DH-GOWT1`
- Track: `01`
- Training frames: `0-54`
- Test frames: `73-91`
- Window size: `5`
- Samples per image: `500`
- Foreground sampling ratio: `0.5`
- Feature mode: enhanced sliding-window features

Implemented feature representation:

- A `5 x 5` sliding pixel window is extracted around each pixel.
- Reflect padding is used at image borders.
- The raw `5 x 5` neighborhood contributes `25` pixel features.
- One additional Sobel-based gradient magnitude feature is added.
- Total feature dimension per pixel: `26`

Model definitions used in code:

- SVM: `SVC(kernel="rbf", probability=True)`
- Logistic Regression: `LogisticRegression(max_iter=1000)`
- Random Forest: `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)`

Evaluation outputs produced by the program:

- Binary mask predictions saved as `mask###.tif`
- Colored qualitative visualizations saved as `colored_mask###.png`
- IoU summary statistics
- SEGMeasure mean Jaccard index
- Per-run report files saved as `latest_evaluation.txt` and `latest_evaluation.json`

## Results

Comparison summary:

| Model | Mean IoU | IoU Std Dev | IoU Min | IoU Max | SEGMeasure Mean Jaccard | Frames Evaluated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM | 0.8896 | 0.0133 | 0.8578 | 0.9021 | 0.8895 | 19 |
| Logistic Regression | 0.8928 | 0.0131 | 0.8607 | 0.9051 | 0.8926 | 19 |
| Random Forest | 0.9037 | 0.0129 | 0.8739 | 0.9178 | 0.9035 | 19 |

Ranking by SEGMeasure:

1. Random Forest: `0.9035`
2. Logistic Regression: `0.8926`
3. SVM: `0.8895`

Ranking by mean IoU:

1. Random Forest: `0.9037`
2. Logistic Regression: `0.8928`
3. SVM: `0.8896`

## Interpretation

Random Forest produced the best overall segmentation quality on the held-out test set. It had the highest mean IoU and the highest SEGMeasure score, while also showing the strongest minimum and maximum IoU values among the three models.

Logistic Regression performed slightly better than SVM in both mean IoU and SEGMeasure. Its accuracy was close to SVM, but consistently a little higher across the summary statistics.

SVM produced the lowest scores of the three models in this comparison, though the gap was not huge. Its results were still competitive, but it did not outperform Logistic Regression or Random Forest on this test split.

## Runtime Observations

Based on observed evaluation behavior:

- SVM took the longest to evaluate.
- Logistic Regression had intermediate evaluation time.
- Random Forest was the quickest to evaluate.

These runtime comments are qualitative because the current saved evaluation reports do not record exact wall-clock time. If exact timing is needed for the final writeup, the evaluation script should be updated to save per-run elapsed time.

Why the runtime likely differed:

- SVM uses an RBF-kernel classifier with probability estimates enabled, which is expensive for dense per-pixel prediction.
- Logistic Regression is computationally simpler and usually faster than kernel SVM during inference.
- Random Forest can parallelize across CPU cores with `n_jobs=-1`, which helps make it much faster in practice in this project.

## Conclusion

For this experiment, Random Forest was the strongest overall model because it achieved the best accuracy and also had the fastest observed evaluation time. Logistic Regression was a solid middle-ground model, with performance slightly above SVM and runtime between SVM and Random Forest. SVM was the slowest to evaluate and also produced the lowest scores among the three tested classical methods.

If one model had to be selected from this comparison, Random Forest would be the most practical choice because it balanced accuracy and speed most effectively.

## Source Files Used For This Report

- `artifacts/results/Fluo-N2DH-GOWT1/01/svm/latest_evaluation.txt`
- `artifacts/results/Fluo-N2DH-GOWT1/01/logreg/latest_evaluation.txt`
- `artifacts/results/Fluo-N2DH-GOWT1/01/rf/latest_evaluation.txt`
- `scripts/train_model.py`
- `scripts/eval_seg.py`
- `src/cell_tracking/models/classical.py`
- `src/cell_tracking/features.py`
