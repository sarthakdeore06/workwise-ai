"""
train_model.py - loads the dataset, trains the Decision Tree, evaluates it on
unseen test data, and saves the model plus the evaluation results.

Run (after generate_dataset.py):  python train_model.py

Steps
-----
1. Load the CSV and validate it (missing values, duplicates, ranges, labels).
2. Split into 80% training and 20% test records (stratified, fixed seed).
3. Choose the tree settings with cross-validation on the TRAINING data only.
4. Compare with two alternative models (for the report, not for selection).
5. Evaluate the final tree ONCE on the untouched test set.
6. Save: model, metrics.json, confusion matrix picture, tree picture,
   and the list of wrongly predicted test records.
"""

import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # draw pictures to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config


# ---------------------------------------------------------------------------
# 1. Load and validate
# ---------------------------------------------------------------------------
def load_and_validate_data() -> pd.DataFrame:
    """Read the CSV, clean small problems, and stop on serious ones."""
    if not config.DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {config.DATA_PATH}. "
            "Run 'python generate_dataset.py' first."
        )

    data = pd.read_csv(config.DATA_PATH)

    # All required columns must be present
    required = [config.ID_COLUMN] + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    missing_columns = [c for c in required if c not in data.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    # Missing values: drop those rows (and tell the user)
    n_before = len(data)
    data = data.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    if len(data) < n_before:
        print(f"Cleaning: dropped {n_before - len(data)} row(s) with missing values.")

    # Duplicate records (same features AND same label): keep only one copy
    n_before = len(data)
    data = data.drop_duplicates(subset=config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    if len(data) < n_before:
        print(f"Cleaning: dropped {n_before - len(data)} duplicate row(s).")

    # Out-of-range values are a real data error: stop instead of silently fixing
    for column in config.FEATURE_COLUMNS:
        low = config.FEATURE_INFO[column]["min"]
        high = config.FEATURE_INFO[column]["max"]
        bad_rows = data[(data[column] < low) | (data[column] > high)]
        if len(bad_rows) > 0:
            raise ValueError(f"{len(bad_rows)} row(s) have '{column}' outside {low}-{high}.")

    # Labels must be valid and every class needs enough examples
    unknown_labels = set(data[config.TARGET_COLUMN]) - set(config.CLASS_LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown labels in target column: {unknown_labels}")
    counts = data[config.TARGET_COLUMN].value_counts()
    for label in config.CLASS_LABELS:
        if counts.get(label, 0) < 20:
            raise ValueError(f"Class '{label}' has fewer than 20 examples; collect more data.")

    return data.reset_index(drop=True)


# ---------------------------------------------------------------------------
# helpers for saving pictures
# ---------------------------------------------------------------------------
def save_confusion_matrix_picture(matrix) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=config.CLASS_LABELS)
    display.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=15)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_MATRIX_PNG, dpi=150)
    plt.close(fig)


def save_tree_picture(model) -> None:
    fig, ax = plt.subplots(figsize=(26, 12))
    plot_tree(model, feature_names=config.FEATURE_COLUMNS,
              class_names=list(model.classes_), filled=True, rounded=True,
              fontsize=7, ax=ax)
    fig.tight_layout()
    fig.savefig(config.TREE_PNG, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main training procedure
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_and_validate_data()
    print(f"Loaded {len(data)} valid records.")

    # X = inputs (features only - no ID, no target), y = answers (labels)
    X = data[config.FEATURE_COLUMNS]
    y = data[config.TARGET_COLUMN]

    # 2. Stratified split: each class keeps the same proportion in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    print(f"Training records: {len(X_train)}   Test records: {len(X_test)}")

    # 3. Choose tree settings using 5-fold cross-validation INSIDE the training set.
    #    The test set is not touched here, so there is no data leakage.
    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=config.RANDOM_SEED),
        param_grid={"max_depth": [2, 3, 4, 5], "min_samples_leaf": [5, 10, 20]},
        cv=cross_validation, scoring="accuracy",
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_   # already re-fitted on the whole training set
    print(f"Best tree settings (from cross-validation): {search.best_params_}")

    # 4. Alternative models, for comparison in the report only.
    #    (The Decision Tree is kept because it is explainable, not because of a test score.)
    candidates = {
        "Decision Tree (chosen)": model,
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED)),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=5, random_state=config.RANDOM_SEED),
    }
    comparison = []
    for name, candidate in candidates.items():
        cv_accuracy = cross_val_score(candidate, X_train, y_train, cv=cross_validation).mean()
        candidate.fit(X_train, y_train)
        test_accuracy = accuracy_score(y_test, candidate.predict(X_test))
        comparison.append({"model": name,
                           "cv_accuracy_train": round(float(cv_accuracy), 4),
                           "test_accuracy": round(float(test_accuracy), 4)})

    # 5. Final evaluation of the chosen tree on unseen test records
    y_pred = model.predict(X_test)
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=config.CLASS_LABELS)
    report = classification_report(y_test, y_pred, labels=config.CLASS_LABELS,
                                   output_dict=True, zero_division=0)

    print(f"\nTrain accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy : {test_accuracy:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(matrix, index=config.CLASS_LABELS, columns=config.CLASS_LABELS).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=config.CLASS_LABELS, zero_division=0))
    print("Model comparison:")
    print(pd.DataFrame(comparison).to_string(index=False))

    # 6. Save everything
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_columns": config.FEATURE_COLUMNS,
        "class_labels": config.CLASS_LABELS,
        "train_feature_means": X_train.mean().round(2).to_dict(),
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, config.MODEL_PATH)

    importances = dict(zip(config.FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))
    metrics = {
        "dataset": {
            "synthetic": True,
            "n_records": int(len(data)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        },
        "best_params": search.best_params_,
        "cv_accuracy_train": round(float(search.best_score_), 4),
        "train_accuracy": round(float(train_accuracy), 4),
        "test_accuracy": round(float(test_accuracy), 4),
        "tree_depth": int(model.get_depth()),
        "tree_leaves": int(model.get_n_leaves()),
        "classification_report": report,
        "confusion_matrix": {"labels": config.CLASS_LABELS, "matrix": matrix.tolist()},
        "feature_importances": importances,
        "model_comparison": comparison,
        "predicted_classes_on_test": sorted(set(y_pred)),
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    save_confusion_matrix_picture(matrix)
    save_tree_picture(model)

    # Wrongly predicted test records, for the error analysis in the report
    wrong = X_test.copy()
    wrong["employee_id"] = data.loc[X_test.index, config.ID_COLUMN]
    wrong["actual"] = y_test
    wrong["predicted"] = y_pred
    wrong = wrong[wrong["actual"] != wrong["predicted"]]
    wrong.to_csv(config.MISCLASSIFIED_PATH, index=False)

    print(f"\nSaved model and results in: {config.MODEL_DIR}")
    print(f"Wrongly predicted test records: {len(wrong)} of {len(y_test)}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)
