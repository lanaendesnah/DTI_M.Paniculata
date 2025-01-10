import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, cohen_kappa_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from joblib import dump, Parallel, delayed

tes = 'AAIndex-PubChem' #ganti sesuai kombinasi fitur

X = pd.read_csv('../../data/4-modelling/1-datainput/aaindex-pubchem_bindingdb.csv') #ganti sesuai kombinasi fitur
y = X.pop('Class')
if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
    X = X.values
if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
    y = y.values

# Models and hyperparameter grids
param_grids = {
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (50, 50)],
        "activation": ["tanh", "relu"],
        "solver": ["adam", "sgd"],
        "alpha": [0.0001, 0.001, 0.01],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
    },
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
    },
}

models = {
    "MLP": MLPClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

# Cross-validation and metrics
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
metrics = [
    ("Accuracy", accuracy_score),
    ("Recall", recall_score),
    ("Precision", precision_score),
    ("F1-Score", f1_score),
    ("AUC", roc_auc_score),
    ("Cohen Kappa", cohen_kappa_score),
]

results = []
roc_curves = []

for model_name, model in models.items():
    print(f"Tuning hyperparameters for {model_name}...")
    param_grid = param_grids[model_name]

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        scoring="accuracy",
        cv=cv,
        random_state=42,
        verbose=1,
        n_jobs=-1,  # Parallelize RandomizedSearchCV
    )
    random_search.fit(X, y)
    best_model = random_search.best_estimator_
    print(f"Best parameters for {model_name}: {random_search.best_params_}")

    # Save Random Search results
    random_search_df = pd.DataFrame(random_search.cv_results_)
    random_search_df.to_csv(f"../../data/4-modelling/result/bindingdb/{tes}/random_search_{model_name.replace(' ', '_').lower()}.csv", index=False)

    # Save the best model
    model_path = f"../../data/4-modelling/result/bindingdb/{tes}/best_model_{model_name.replace(' ', '_').lower()}.joblib"
    dump(best_model, model_path)
    print(f"Best model for {model_name} saved to {model_path}")

    print(f"Evaluating {model_name}...")
    metric_scores = {metric[0]: [] for metric in metrics}
    all_roc_data = []

    # Parallel evaluation for each fold
    def evaluate_fold(train_idx, test_idx):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba")
            else None
        )

        fold_results = {}
        for metric_name, metric_func in metrics:
            if metric_name == "AUC" and y_proba is not None:
                score = metric_func(y_test, y_proba)
            elif metric_name != "AUC":
                score = metric_func(y_test, y_pred)
            else:
                score = np.nan
            fold_results[metric_name] = score

        # Compute ROC curve
        roc_data = {}
        if y_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            roc_data = {"FPR": fpr, "TPR": tpr, "Thresholds": thresholds}

        return fold_results, roc_data

    results_per_fold = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(train_idx, test_idx) for train_idx, test_idx in cv.split(X, y)
    )

    for fold_results, roc_data in results_per_fold:
        for metric_name, score in fold_results.items():
            metric_scores[metric_name].append(score)
        if roc_data:
            all_roc_data.append(roc_data)

    # Store mean and std metrics
    result = {
        "Model": model_name,
        **{f"Mean {metric[0]}": np.mean(scores) for metric, scores in zip(metrics, metric_scores.values())},
        **{f"Std {metric[0]}": np.std(scores) for metric, scores in zip(metrics, metric_scores.values())},
    }
    results.append(result)

    # Save ROC curve data
    roc_data_combined = {
        "FPR": [],
        "TPR": [],
        "Thresholds": [],
        "Fold": [],
    }
    for fold_idx, roc_data in enumerate(all_roc_data, start=1):
        roc_data_combined["FPR"].extend(roc_data["FPR"])
        roc_data_combined["TPR"].extend(roc_data["TPR"])
        roc_data_combined["Thresholds"].extend(roc_data["Thresholds"])
        roc_data_combined["Fold"].extend([fold_idx] * len(roc_data["FPR"]))

    roc_df = pd.DataFrame(roc_data_combined)
    roc_df.to_csv(f"../../data/4-modelling/result/bindingdb/{tes}/roc_curve_{model_name.replace(' ', '_').lower()}.csv", index=False)

# Save summary results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(f"../../data/4-modelling/result/bindingdb/{tes}/evaluation_metrics.csv", index=False)

print("Evaluation completed. All results saved in the 'result' folder.")
