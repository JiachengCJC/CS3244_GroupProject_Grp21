# CS3244_GroupProject_Grp21

## Wuting Branch Submission (Classical ML)

This branch contains Wu Ting's classical machine learning implementation and experiment artifacts in:

- `src/wuting_classical_ml/classical_ml.py`
- `src/wuting_classical_ml/requirements.txt`
- `src/wuting_classical_ml/artifacts/classical_ml/classical_summary.csv`
- `src/wuting_classical_ml/artifacts/classical_ml/classical_summary.json`
- `src/wuting_classical_ml/artifacts/classical_ml/classical_complexity.csv`

## Evaluation Setup

- Dataset: Fashion-MNIST from OpenML (`fetch_openml`)
- Models: Logistic Regression, SVM (RBF), Random Forest
- CV: 5-fold stratified CV (`random_state=0`)
- Metrics: CV accuracy, test accuracy, top-3/top-5 accuracy, macro precision/recall/F1, training/inference time
- Additional analysis: No-PCA vs PCA ablation, confusion matrices, misclassified sample plots, model complexity table

Current artifact numbers in this branch are generated from a reproducible stratified subset run:

```bash
python3 src/wuting_classical_ml/classical_ml.py \
  --max-train-samples 1200 \
  --max-test-samples 400 \
  --output-dir src/wuting_classical_ml/artifacts/classical_ml
```

## Results (from `classical_summary.csv`)

### PCA Setting

| Model | CV Accuracy (mean ± std) | Test Accuracy | Top-3 | Top-5 | Macro F1 | Train (s) | Infer (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8000 ± 0.0201 | 0.7550 | 0.9575 | 0.9875 | 0.7569 | 14.35 | 0.003 |
| SVM (RBF) | 0.8183 ± 0.0125 | 0.8050 | 0.9825 | 0.9975 | 0.8060 | 8.85 | 0.029 |
| Random Forest | 0.8100 ± 0.0198 | 0.7825 | 0.9775 | 0.9925 | 0.7786 | 10.48 | 0.248 |

### No-PCA vs PCA Ablation

| Model | CV No-PCA | CV PCA | Test No-PCA | Test PCA |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8025 ± 0.0217 | 0.8000 ± 0.0201 | 0.7500 | 0.7550 |
| SVM (RBF) | N/A (infeasible) | 0.8183 ± 0.0125 | N/A (infeasible) | 0.8050 |
| Random Forest | 0.8200 ± 0.0213 | 0.8100 ± 0.0198 | 0.7850 | 0.7825 |


## Notes

To keep branch size small and avoid unnecessary binary diffs, only summary artifacts (CSV/JSON) are committed. Figures can be regenerated locally by running the script.
