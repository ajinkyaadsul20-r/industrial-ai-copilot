# Industrial AI Copilot
### Explainable Predictive Maintenance with Model Evaluation and What-if Analysis

An end-to-end tabular machine-learning portfolio project for equipment-failure classification. Includes CSV upload, data-quality checks, two model choices, held-out evaluation, confusion matrix, permutation importance, interactive what-if predictions, and downloadable model/run metadata.

> **Important:** This is an educational prototype, not a validated industrial safety system. Synthetic demo data is included only to make the app usable immediately. Do not report demo metrics as real-world accuracy.

## Features
- CSV upload and selectable binary target (`0` = no failure, `1` = failure)
- Data preview, missingness and unique-value report
- Random Forest and Histogram Gradient Boosting
- Accuracy, precision, recall, F1, ROC-AUC, average precision and confusion matrix
- Permutation feature importance
- Interactive what-if prediction form
- Downloadable fitted pipeline and JSON run manifest
- No API key required

## Run locally
Python 3.10 or 3.11 recommended.

**Windows / VS Code terminal**
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Dataset
The app starts with generated synthetic data. For a credible portfolio evaluation, consider the [AI4I 2020 Predictive Maintenance Dataset from UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). Read the dataset documentation and terms, cite the source, inspect target labels, and avoid leakage/identifier columns. Select the correct binary target in the sidebar; convert it to 0/1 if needed.

## Evaluation
The app uses a stratified train/test split and a scikit-learn pipeline for imputing missing values and one-hot encoding categorical features. Review recall and false negatives, not just accuracy. For a serious study, add cross-validation and temporal validation where timestamps exist.

- **Precision:** Of predicted failures, fraction that are true failures.
- **Recall:** Of actual failures, fraction found by the model.
- **F1:** Harmonic mean of precision and recall.
- **ROC-AUC / average precision:** Ranking metrics for binary classification.
- **Permutation importance:** How model score changes when a feature is shuffled; not causal evidence.

## Repository structure
```text
industrial-ai-copilot/
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── CONTRIBUTING.md
├── docs/
│   ├── EXPERIMENTS.md
│   ├── MODEL_CARD.md
│   └── screenshots/README.md
└── tests/test_repository.py
```

## Limitations
- Synthetic demo data is unsuitable for real-world conclusions.
- Random hold-out metrics may not represent future operating conditions.
- Scores may be miscalibrated and are not guaranteed failure probabilities.
- Feature importance is descriptive, not causal.
- Real deployment requires representative data, domain validation, monitoring, security review and human oversight.

## Author
Replace with your name and GitHub profile. Describe only work you personally completed and report only results you have verified.
