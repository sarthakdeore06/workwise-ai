"""
test_cases.py - runs the planned test cases and prints the ACTUAL results.

Run (after train_model.py):  python test_cases.py

It uses the same functions as the web app (predictor.py and recommender.py),
so a pass here means the app logic behaves as expected. It does not test the
browser screen itself; do that by hand and take screenshots.

The expected results were written BEFORE running, based on common sense about
the scenarios. If the model disagrees, the case is reported as FAIL - we do not
change the expectations to make them pass.
"""

import copy

import pandas as pd

import config
import predictor
import recommender

SAMPLES = config.SAMPLE_EMPLOYEES
bundle = predictor.load_model_bundle()


def predict_case(values, expected_category):
    """Valid input: check that the predicted category equals the expected one."""
    errors = predictor.validate_inputs(values)
    if errors:
        return f"Rejected: {errors[0]}", False
    category, _ = predictor.predict_category(bundle, values)
    return f"Predicted: {category}", category == expected_category


def accepted_case(values):
    """Valid input: check only that it is accepted and a valid category comes back."""
    errors = predictor.validate_inputs(values)
    if errors:
        return f"Rejected: {errors[0]}", False
    category, _ = predictor.predict_category(bundle, values)
    return f"Accepted, predicted: {category}", category in config.CLASS_LABELS


def rejected_case(values):
    """Invalid input: check that validation reports at least one error."""
    errors = predictor.validate_inputs(values)
    if errors:
        return f"Rejected: {errors[0]}", True
    return "Accepted (should have been rejected)", False


def with_change(base_name, **changes):
    values = copy.deepcopy(SAMPLES[base_name])
    values.update(changes)
    return values


def all_minimum():
    return {n: config.FEATURE_INFO[n]["min"] for n in config.FEATURE_COLUMNS}


def all_maximum():
    return {n: config.FEATURE_INFO[n]["max"] for n in config.FEATURE_COLUMNS}


def best_extreme():
    return {"task_completion_rate": 100.0, "on_time_delivery_rate": 100.0, "error_rate": 0.0,
            "attendance_pct": 100.0, "training_hours": 40.0, "peer_collaboration_score": 5.0}


def worst_extreme():
    return {"task_completion_rate": 40.0, "on_time_delivery_rate": 30.0, "error_rate": 20.0,
            "attendance_pct": 60.0, "training_hours": 0.0, "peer_collaboration_score": 1.0}


def missing_field_case():
    values = copy.deepcopy(SAMPLES["Typical employee"])
    del values["attendance_pct"]
    return rejected_case(values)


def mixed_profile_case():
    values = SAMPLES["Fast but often late"]
    errors = predictor.validate_inputs(values)
    if errors:
        return f"Rejected: {errors[0]}", False
    category, _ = predictor.predict_category(bundle, values)
    flagged = [item["indicator"] for item in recommender.get_recommendations(values)["improvements"]]
    return f"Predicted: {category}; suggestions for: {flagged}", ("Timeliness" in flagged)


def missing_model_case():
    try:
        predictor.load_model_bundle(config.MODEL_DIR / "does_not_exist.joblib")
    except FileNotFoundError as error:
        return f"FileNotFoundError: {str(error)[:60]}...", "train_model.py" in str(error)
    return "No error raised", False


def strong_employee_recommendation_case():
    advice = recommender.get_recommendations(SAMPLES["High performer"])
    return (f"{len(advice['improvements'])} improvement(s), {len(advice['strengths'])} strength(s)",
            len(advice["improvements"]) == 0 and len(advice["strengths"]) > 0)


def struggling_employee_recommendation_case():
    advice = recommender.get_recommendations(SAMPLES["Struggling employee"])
    return f"{len(advice['improvements'])} improvement area(s) flagged", len(advice["improvements"]) == 6


def all_classes_case():
    data = pd.read_csv(config.DATA_PATH)
    predictions = bundle["model"].predict(data[config.FEATURE_COLUMNS])
    found = sorted(set(predictions))
    return f"Classes predicted: {found}", set(found) == set(config.CLASS_LABELS)


def explanation_case():
    steps = predictor.explain_decision_path(bundle, SAMPLES["Typical employee"])
    depth = bundle["model"].get_depth()
    ok = 1 <= len(steps) <= depth and all(s["feature"] in config.FEATURE_COLUMNS for s in steps)
    return f"{len(steps)} question(s) shown (tree depth {depth})", ok


TEST_CASES = [
    ("TC01", "Typical employee (normal input)", "Meeting Expectations",
     lambda: predict_case(SAMPLES["Typical employee"], "Meeting Expectations")),
    ("TC02", "Struggling employee (normal input)", "Needs Support",
     lambda: predict_case(SAMPLES["Struggling employee"], "Needs Support")),
    ("TC03", "High performer (normal input)", "Exceeding Expectations",
     lambda: predict_case(SAMPLES["High performer"], "Exceeding Expectations")),
    ("TC04", "Fast but often late: high completion, 55% on time",
     "Valid category AND a timeliness suggestion", mixed_profile_case),
    ("TC05", "Boundary: every feature at its minimum allowed value",
     "Accepted; a valid category is returned", lambda: accepted_case(all_minimum())),
    ("TC06", "Boundary: every feature at its maximum allowed value",
     "Accepted; a valid category is returned", lambda: accepted_case(all_maximum())),
    ("TC07", "Best possible extreme (100/100/0 errors/100/40 h/5.0)", "Exceeding Expectations",
     lambda: predict_case(best_extreme(), "Exceeding Expectations")),
    ("TC08", "Worst possible extreme (40/30/20 errors/60/0 h/1.0)", "Needs Support",
     lambda: predict_case(worst_extreme(), "Needs Support")),
    ("TC09", "Invalid: task completion 100.1 (above maximum)", "Rejected with error message",
     lambda: rejected_case(with_change("Typical employee", task_completion_rate=100.1))),
    ("TC10", "Invalid: negative error rate (-1)", "Rejected with error message",
     lambda: rejected_case(with_change("Typical employee", error_rate=-1))),
    ("TC11", "Invalid: text 'abc' in attendance", "Rejected with error message",
     lambda: rejected_case(with_change("Typical employee", attendance_pct="abc"))),
    ("TC12", "Invalid: attendance field missing", "Rejected with error message", missing_field_case),
    ("TC13", "Invalid: NaN in training hours", "Rejected with error message",
     lambda: rejected_case(with_change("Typical employee", training_hours=float("nan")))),
    ("TC14", "Model file missing", "Friendly FileNotFoundError telling user to run train_model.py",
     missing_model_case),
    ("TC15", "Recommendations for high performer", "No improvement areas; strengths listed",
     strong_employee_recommendation_case),
    ("TC16", "Recommendations for struggling employee", "All 6 indicators flagged",
     struggling_employee_recommendation_case),
    ("TC17", "Coverage: model predicts all 3 classes on the dataset", "All 3 classes appear",
     all_classes_case),
    ("TC18", "Explanation for typical employee", "1 to tree-depth questions, valid feature names",
     explanation_case),
]


def main():
    rows = []
    for case_id, scenario, expected, run in TEST_CASES:
        actual, passed = run()
        rows.append({"Test ID": case_id, "Scenario": scenario, "Expected": expected,
                     "Actual": actual, "Result": "PASS" if passed else "FAIL"})
        print(f"{case_id} [{'PASS' if passed else 'FAIL'}] {scenario}\n      expected: {expected}\n      actual  : {actual}")

    results = pd.DataFrame(rows)
    results.to_csv(config.TEST_RESULTS_PATH, index=False)
    n_pass = int((results["Result"] == "PASS").sum())
    print(f"\n{n_pass} of {len(results)} test cases passed. Saved to {config.TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
