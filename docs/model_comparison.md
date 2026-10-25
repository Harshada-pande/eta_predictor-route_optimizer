# Indian Food Delivery ETA Model Comparison

## Dataset

- Training rows: **33562**
- Internal validation rows: **6713**
- Held-out test rows: **8391**
- Input features: **27 features + 1 target = 28 columns**
- Target: **Time_taken_min**
- The held-out test set was kept separate from model selection.

## Model Selection

Only the two ML models — **Random Forest** and **XGBoost** — were eligible for final model selection.
The selected model was **XGBoost**, with the lowest validation MAE of **3.1912 minutes**.

Baselines were evaluated only to provide reference performance and were not eligible to become the final model.

## Validation Results

| Model | MAE (min) | RMSE (min) | R² | Variance Explained | Delay Accuracy | Delay F1 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 3.1912 | 4.0079 | 0.8209 | 82.09% | 0.9307 | 0.8684 |
| RandomForest | 3.2492 | 4.0964 | 0.8129 | 81.29% | 0.9283 | 0.8657 |
| DistanceOnlyLinear | 7.2968 | 8.9299 | 0.1110 | 11.10% | 0.7269 | 0.0508 |
| MedianBaseline | 7.6349 | 9.4749 | -0.0008 | -0.08% | 0.7290 | 0.0000 |
| MeanBaseline | 7.6624 | 9.4713 | -0.0000 | -0.00% | 0.7290 | 0.0000 |

## Final Held-Out Test Performance

- Model: **XGBoost**
- MAE: **3.0852 minutes**
- MSE: **14.9830**
- RMSE: **3.8708 minutes**
- R²: **0.8271**
- Variance explained: **82.71%**

### Delay-Risk Performance

- Delay threshold: **32.0000 minutes**
- Accuracy: **0.9317**
- Error rate: **0.0683**
- Precision: **0.9088**
- Recall: **0.8371**
- F1 score: **0.8714**

## Metric Interpretation

- **MAE** represents the average absolute ETA prediction error in minutes.
- **RMSE** penalizes larger prediction errors more strongly than MAE.
- **R²** measures the proportion of variance in delivery time explained by the model.
- **R² is not classification accuracy.** For example, R² = 0.8271 means the model explains approximately 82.71% of the variance in delivery time.
- **Delay accuracy, precision, recall and F1** belong to the separate delay-risk classification layer.

## Baseline Purpose

The baseline models establish simple reference points against which the machine-learning models can be evaluated.
- Mean baseline: predicts the average delivery time.
- Median baseline: predicts the median delivery time.
- Distance-only linear regression: estimates delivery time using only distance.

## Final Model

The final model is **XGBoost**.
It was selected using internal validation data and then retrained on the complete training dataset before the final held-out evaluation.