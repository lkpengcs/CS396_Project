# baseline_models.py
import json
import joblib
import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import scipy.sparse
# Add parent directory to path to import metrics
sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import evaluate_metrics, calculate_performance_metrics, prepare_groups


if __name__ == "__main__":
    # Get parent directory path
    parent_dir = Path(__file__).parent.parent

    # Load raw splits
    X_train_raw = pd.read_csv(parent_dir / "X_train_raw.csv")
    y_train = pd.read_csv(parent_dir / "y_train.csv").squeeze()

    X_test_raw = pd.read_csv(parent_dir / "X_test_raw.csv")
    y_test = pd.read_csv(parent_dir / "y_test.csv").squeeze()

    # Load encoder
    preprocessor = joblib.load(parent_dir / "preprocessor.pkl")

    # Protected attributes for fairness evaluation
    protected_attrs = ["sex", "race", "age", "marital.status", "education", "relationship"]

    # Define models
    models = [
        (LogisticRegression(max_iter=3000), "logistic_regression"),
        (RandomForestClassifier(n_estimators=200), "random_forest"),
        (GradientBoostingClassifier(), "gradient_boosting")
    ]

    all_results = []

    X_train = preprocessor.transform(X_train_raw)

    # Identify numeric and categorical
    num_cols = X_train_raw.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train_raw.select_dtypes(include=["object"]).columns.tolist()

    # Get OHE feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(cat_cols).tolist()

    # Combine
    feature_names = num_cols + cat_feature_names

    # Save
    with open(parent_dir / "baseline" / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    X_test = preprocessor.transform(X_test_raw)
    # Save transformed X_test
    transformed_X_test_path = parent_dir / "baseline" / "transformed_X_test.npz"
    scipy.sparse.save_npz(transformed_X_test_path, X_test)

    for model, model_name in models:
        print(f"\n{'='*60}")
        print(f"Running {model_name}...")
        print(f"{'='*60}")
        
        # Train model once (performance metrics are the same regardless of protected attr)

        model.fit(X_train, y_train)
        # Save model files
        model_dir = parent_dir / "baseline" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / f"{model_name}.pkl")
        # Get predictions once
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        # Save predictions and probabilities
        predictions_path = model_dir / f"{model_name}_predictions.csv"
        predictions_df = pd.DataFrame({
            "y_pred": y_pred,
            "y_prob": y_prob
        })
        predictions_df.to_csv(predictions_path, index=False)
        
        # Calculate performance metrics (same for all protected attributes)
        performance = calculate_performance_metrics(y_test.values, y_pred, y_prob)
        
        # Calculate fairness metrics for each protected attribute
        fairness = {}
        for protected_attr in protected_attrs:
            print(f"  Evaluating with protected attribute: {protected_attr}")
            groups = prepare_groups(X_test_raw, protected_attr)
            fairness_results = evaluate_metrics(
                y_true=y_test.values, 
                y_pred=y_pred, 
                y_prob=y_prob, 
                groups=groups
            )
            
            # Extract only fairness-related metrics
            fairness[protected_attr] = {
                "dpd": fairness_results["dpd"],
                "eod": fairness_results["eod"],
                "fprd": fairness_results["fprd"],
                "calibration": fairness_results["calibration"]
            }
        
        # Structure results
        model_result = {
            "model_name": model_name,
            "performance": performance,
            "fairness": fairness
        }
        all_results.append(model_result)

    # Save results
    output_path = Path(__file__).parent / "baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"\n✔ All baseline metrics saved to {output_path}")