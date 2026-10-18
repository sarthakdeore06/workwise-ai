"""
predictor.py - everything needed to USE the trained model (inference).

app.py and test_cases.py both import these functions, so the web page and the
tests always behave identically. Nothing here trains or changes the model.
"""

import math

import joblib
import numpy as np
import pandas as pd

import config


def load_model_bundle(path=config.MODEL_PATH) -> dict:
    """Load the saved model bundle, or explain clearly how to create it."""
    if not path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run 'python generate_dataset.py' and then "
            "'python train_model.py' before starting the app."
        )
    return joblib.load(path)


def validate_inputs(values: dict) -> list:
    """Check the six inputs. Returns a list of error messages (empty = all valid)."""
    errors = []
    for name in config.FEATURE_COLUMNS:
        info = config.FEATURE_INFO[name]
        label = info["label"]

        if name not in values or values[name] is None:
            errors.append(f"{label} is missing.")
            continue

        raw = values[name]
        if isinstance(raw, bool):
            errors.append(f"{label} must be a number (got {raw!r}).")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{label} must be a number (got {raw!r}).")
            continue

        if math.isnan(number) or math.isinf(number):
            errors.append(f"{label} must be a finite number.")
        elif not info["min"] <= number <= info["max"]:
            errors.append(f"{label} must be between {info['min']:g} and "
                          f"{info['max']:g} (got {number:g}).")
    return errors


def make_input_frame(values: dict) -> pd.DataFrame:
    """Turn the six values into a one-row table with the columns the model expects."""
    row = {name: float(values[name]) for name in config.FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=config.FEATURE_COLUMNS)  # same order as training


def predict_category(bundle: dict, values: dict):
    """Return (predicted category, {class: probability}) for one employee."""
    model = bundle["model"]
    frame = make_input_frame(values)
    category = model.predict(frame)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(frame)[0]))
    # Show probabilities in the natural order Needs Support -> Meeting -> Exceeding
    ordered = {label: float(probabilities.get(label, 0.0)) for label in config.CLASS_LABELS}
    return category, ordered


def explain_decision_path(bundle: dict, values: dict) -> list:
    """
    List the yes/no questions the tree asked for THIS employee, in order.

    This describes how the model reached its answer using patterns found in
    the training data. It shows association, not cause and effect.
    """
    model = bundle["model"]
    tree = model.tree_
    frame = make_input_frame(values)

    # scikit-learn compares float32 numbers internally, so we do the same
    x = frame.to_numpy(dtype=np.float32)

    # decision_path gives the ids of all tree nodes this record passes through
    node_indicator = model.decision_path(frame)
    node_ids = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    steps = []
    for node_id in node_ids:
        is_leaf = tree.children_left[node_id] == tree.children_right[node_id]
        if is_leaf:
            continue  # leaves hold the answer; they ask no question
        feature_index = int(tree.feature[node_id])
        feature_name = bundle["feature_columns"][feature_index]
        threshold = float(tree.threshold[node_id])
        value = float(x[0, feature_index])
        goes_left = value <= threshold
        label = config.FEATURE_INFO[feature_name]["label"]
        steps.append({
            "feature": feature_name,
            "label": label,
            "value": float(frame.iloc[0][feature_name]),
            "threshold": threshold,
            "answer": "yes" if goes_left else "no",
            "text": (f"Is {label} <= {threshold:.1f}?  "
                     f"Your value is {float(frame.iloc[0][feature_name]):g}, so the answer is "
                     f"{'YES' if goes_left else 'NO'}."),
        })
    return steps


def get_feature_importances(bundle: dict) -> dict:
    """Share of the tree's splitting power used by each feature (sums to 1)."""
    importances = dict(zip(bundle["feature_columns"], bundle["model"].feature_importances_))
    return dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))
