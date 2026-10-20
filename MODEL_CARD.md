# Model Card

## Intended use
Educational exploration of binary equipment-failure classification on tabular sensor data.

## Out-of-scope use
Not validated for safety-critical decisions, autonomous shutdown, or unsupervised industrial deployment.

## Models
Random Forest and Histogram Gradient Boosting. Inputs are user-selected columns; target is a binary failure label.

## Data
The app includes synthetic demo data for workflow testing. Real evaluation requires documented, representative labeled data.

## Limitations
Dataset shift, noisy labels, leakage, imbalance and temporal effects can reduce performance. Probabilities may require calibration. Permutation importance is not causal. One random split is not sufficient for deployment readiness.

## Human oversight
Use qualified maintenance personnel and established inspection procedures. Never rely on a model score alone to determine whether equipment is safe.
