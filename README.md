# Thyroid ML Assignment — Project Root

This repository contains code and artifacts for a thyroid disease classification assignment.

Structure
- `thyroid.csv` — raw dataset (used for training and mapping labels)
- `ML_code.ipynb` — original notebook (EDA, preprocessing, multiclass models)
- `app.py` — Streamlit UI


Quick start (recommended workflow)
1. Create a virtual environment (recommended):

   python3 -m venv .venv
   source .venv/bin/activate

2. Install dependencies for the project:

   pip install -r requirements.txt

3. Train the binary model (this reads `thyroid.csv` from project root ):

   python3 binary_app/train_and_save_model.py

4. Run the binary Streamlit app:

   streamlit run app.py

Notes
- The binary trainer maps the original multiclass target to a binary target: `'-'` → Normal (0), any other label → Disease (1). This reduces the severe class imbalance impact and provides a focused disease detector.

- If you prefer to retrain or adjust mappings (e.g., group specific labels into disease subtypes), edit `binary_app/train_and_save_model.py`.



# Problem statement

This project aims to build and evaluate machine learning models to detect thyroid disease from clinical/laboratory features. The goal is to compare six models' predictive performance (binary classification: Normal vs Disease) and choose the best performing model(s) based on multiple metrics.

# Dataset description

- Dataset file: `thyroid.csv`
- Task: Binary classification (Normal vs Disease). The original multiclass labels were mapped to binary with `'-'` → Normal (0), others → Disease (1).
- Features: clinical and laboratory measurements (see `ML_code.ipynb` for full feature list and preprocessing steps).
- Preprocessing summary: missing-value imputation, scaling (where required), categorical encoding, train/test split (reported in notebook).

# Models used

Make a Comparison Table with the evaluation metrics calculated for all six models:

| ML Model Name      | Accuracy | AUC  | Precision | Recall | F1   | MCC  |
|--------------------|:--------:|:----:|:---------:|:------:|:----:|:----:|
| Logistic Regression|  0.837   | N/A  |  0.803    | 0.837  | 0.810| 0.600|
| Decision Tree      |  0.916   | N/A  |  0.911    | 0.916  | 0.913| 0.814|
| k-NN               |  0.831   | N/A  |  0.813    | 0.831  | 0.807| 0.583|
| Naive Bayes        |  0.050   | N/A  |  0.796    | 0.050  | 0.033| 0.072|
| Random Forest      |  0.919   | N/A  |  0.908    | 0.919  | 0.912| 0.820|
| XGBoost            |  0.920   | N/A  |  0.909    | 0.920  | 0.914| 0.822|

- Notes: Values rounded to 3 decimals. Use the same test split for all model evaluations and report how AUC was computed (e.g., ROC AUC on test set probabilities).

- Suggested evaluation pipeline:
  - Use stratified train/test split (e.g., 80/20).
  - Fit models on training set; obtain predicted probabilities and labels on test set.
  - Compute Accuracy, ROC AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC).

- Example code references are in `ML_code.ipynb`.

# Observations on the performance of each model on the chosen dataset.

| ML Model Name      | Observation about model performance |
|--------------------|--------------------------------------|
| Logistic Regression| Good baseline with balanced precision and recall; interpretable coefficients; may underfit complex non-linear patterns. |
| Decision Tree      | High accuracy and strong precision/recall on this split; can overfit if not pruned and is less stable than ensembles. |
| k-NN               | Competitive baseline, sensitive to feature scaling and k choice; slightly lower overall than tree-based ensembles. |
| Naive Bayes        | Very low accuracy here despite high precision for a predicted class — likely affected by class imbalance or violated independence assumptions; not suitable without further preprocessing. |
| Random Forest      | Robust ensemble with strong accuracy and MCC; good generalization and stability versus single trees. |
| XGBoost            | Top performer (marginally better than Random Forest) with highest MCC; benefits from boosting of weak learners and hyperparameter tuning. |
---
Final Link: https://thyroidclassification24x7.streamlit.app/
