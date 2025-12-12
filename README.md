# CS396 Project — Fairness, Bias Mitigation, and Analysis

This repo trains baseline income classifiers on Adult Census-style tabular data, applies multiple fairness/bias-mitigation methods, and produces analysis artifacts (fairness metrics, error asymmetry, calibration, counterfactual flip rates, and figures).
#### Trained model .pkl files are too large to upload to github so they are saved here: https://drive.google.com/drive/folders/16NorZ6R5xVCw0NejPqKxSxXiKUFPo1rv?usp=sharing 

## Project layout

- `preprocess.py`  
  Builds the preprocessing pipeline (e.g., one-hot encoding + scaling) and saves `preprocessor.pkl`.

- `split.py`  
  Splits data into train/val/test CSVs:
  `X_train_raw.csv`, `X_val_raw.csv`, `X_test_raw.csv`,
  `y_train.csv`, `y_val.csv`, `y_test.csv`.

- `baseline/` 
  Baseline models + outputs.
  - `baseline_results.json`
  - `models/*_predictions.csv` (baseline predictions: `y_pred`, `y_prob`)
  - `feature_names.json`
  - `transformed_X_test.npz` (cached test features)

- `advanced/`
  advanced model (stretched deliverable)
- `fairness_mitigation.py`  
  Trains bias-mitigated models and saves them to `saved_models/*.pkl`.

- `mitigated/mitigated.py`  
  Evaluates mitigated models, writes `mitigated/mitigated_results.json`,
  and saves predictions to `mitigated/models/*_predictions.csv`.  

- `error_analysis/`  
  Error-asymmetry + feature-importance analysis scripts and CSV outputs.

- `counterfactual_analysis/`  
  Counterfactual flip-rate analysis scripts and outputs.

- `figures/`, `figures_mitigated/`, etc.  
  Generated plots.

use environment.yml file to set up environment
