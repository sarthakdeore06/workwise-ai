"""
app.py - the Streamlit web interface for WorkWise AI.

Run:  streamlit run app.py

This file only DISPLAYS things. The prediction logic lives in predictor.py,
the suggestions in recommender.py, and the training in train_model.py.
"""

import json

import pandas as pd
import streamlit as st

import config
import predictor
import recommender

st.set_page_config(page_title="WorkWise AI", layout="wide")


# ---------------------------------------------------------------------------
# Loading helpers (cached so the files are read only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_bundle():
    return predictor.load_model_bundle()


def load_metrics():
    if not config.METRICS_PATH.exists():
        return None
    with open(config.METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------------------------
# Page 1: Predict
# ---------------------------------------------------------------------------
def show_predict_page(bundle):
    st.title("WorkWise AI - Employee Performance Insight and Support")
    st.warning(
        "Educational prototype. It uses SYNTHETIC (fictional) data and is not a real HR tool. "
        "Its output must never be used alone to hire, fire, promote or penalise anyone. "
        "Human review and context are essential."
    )

    left, right = st.columns([1, 1.4])

    # ---- inputs -----------------------------------------------------------
    with left:
        st.subheader("1. Employee indicators")
        preset_options = list(config.SAMPLE_EMPLOYEES.keys()) + ["Custom (start from typical)"]
        preset = st.selectbox("Start from a sample (fictional) employee", preset_options)
        preset_values = config.SAMPLE_EMPLOYEES.get(preset, config.SAMPLE_EMPLOYEES["Typical employee"])

        values = {}
        for name in config.FEATURE_COLUMNS:
            info = config.FEATURE_INFO[name]
            values[name] = st.number_input(
                info["label"],
                min_value=info["min"], max_value=info["max"],
                value=float(preset_values[name]), step=info["step"],
                help=info["help"],
                key=f"{name}_{preset}",   # new key per preset -> boxes reset to the preset
            )

    # ---- validation -------------------------------------------------------
    errors = predictor.validate_inputs(values)
    if errors:
        with right:
            for message in errors:
                st.error(message)
        return

    # ---- prediction -------------------------------------------------------
    category, probabilities = predictor.predict_category(bundle, values)

    with right:
        st.subheader("2. AI prediction")
        if category == "Needs Support":
            st.warning(f"Predicted category: **{category}**")
        elif category == "Meeting Expectations":
            st.info(f"Predicted category: **{category}**")
        else:
            st.success(f"Predicted category: **{category}**")
        st.caption(
            "The bars show the share of similar TRAINING records (in the final tree leaf) "
            "that had each label. This is not a guarantee or a calibrated confidence."
        )
        probability_table = pd.DataFrame({"share of similar training records": probabilities})
        st.bar_chart(probability_table)

    # ---- details in tabs --------------------------------------------------
    tab_why, tab_help, tab_compare = st.tabs(
        ["Why this prediction?", "Improvement suggestions", "Compared with training data"]
    )

    with tab_why:
        st.write("The Decision Tree asked these questions about this employee, in order:")
        for number, step in enumerate(predictor.explain_decision_path(bundle, values), start=1):
            st.write(f"{number}. {step['text']}")
        st.caption(
            "This shows how the model reached its answer using patterns in the training data. "
            "It shows association, not cause and effect."
        )
        st.write("Overall feature importance of the trained tree (all employees):")
        importance_table = pd.DataFrame(
            {"importance": predictor.get_feature_importances(bundle)}
        )
        importance_table.index = [config.FEATURE_INFO[n]["label"] for n in importance_table.index]
        st.bar_chart(importance_table)

    with tab_help:
        st.caption(
            "These suggestions come from simple threshold rules in recommender.py. "
            "They are practical guidance, NOT output of the machine-learning model."
        )
        advice = recommender.get_recommendations(values)
        if advice["improvements"]:
            for item in advice["improvements"]:
                st.markdown(f"**{item['indicator']}** - {item['observation']}")
                st.write(f"Suggestion: {item['suggestion']}")
        else:
            st.success("No indicator is below the suggestion thresholds.")
        if advice["strengths"]:
            st.markdown("**Visible strengths**")
            for strength in advice["strengths"]:
                st.write(f"- {strength}")

    with tab_compare:
        rows = []
        for name in config.FEATURE_COLUMNS:
            rows.append({
                "Indicator": config.FEATURE_INFO[name]["label"],
                "Entered value": values[name],
                "Training-data average": bundle["train_feature_means"][name],
            })
        st.table(pd.DataFrame(rows).set_index("Indicator"))


# ---------------------------------------------------------------------------
# Page 2: Model evaluation
# ---------------------------------------------------------------------------
def show_evaluation_page():
    st.title("Model Evaluation")
    metrics = load_metrics()
    if metrics is None:
        st.error("metrics.json not found. Run 'python train_model.py' first.")
        return

    st.info(
        "Results below come from the held-out TEST set (records the model never saw during "
        "training). The data is synthetic, so these numbers show that the pipeline works; "
        "they do not prove real-world accuracy."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test accuracy", f"{metrics['test_accuracy'] * 100:.1f}%")
    col2.metric("Training accuracy", f"{metrics['train_accuracy'] * 100:.1f}%")
    col3.metric("Cross-validation accuracy (train)", f"{metrics['cv_accuracy_train'] * 100:.1f}%")
    col4.metric("Tree depth / leaves", f"{metrics['tree_depth']} / {metrics['tree_leaves']}")

    dataset = metrics["dataset"]
    st.write(f"Records: {dataset['n_records']} total, {dataset['n_train']} for training, "
             f"{dataset['n_test']} for testing. Best settings: {metrics['best_params']}.")

    st.subheader("Confusion matrix (rows = actual, columns = predicted)")
    labels = metrics["confusion_matrix"]["labels"]
    matrix = pd.DataFrame(metrics["confusion_matrix"]["matrix"], index=labels, columns=labels)
    st.table(matrix)
    if config.CONFUSION_MATRIX_PNG.exists():
        st.image(str(config.CONFUSION_MATRIX_PNG), width=480)

    st.subheader("Precision, recall and F1-score per class")
    report = pd.DataFrame(metrics["classification_report"]).T
    report = report.loc[labels + ["macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]]
    report["support"] = report["support"].astype(int)   # show record counts as whole numbers
    st.table(report.round(3))

    st.subheader("Comparison with alternative models")
    st.table(pd.DataFrame(metrics["model_comparison"]).set_index("model"))
    st.caption("The Decision Tree is kept because its decisions can be explained step by step.")

    if config.TREE_PNG.exists():
        with st.expander("Show the trained decision tree"):
            st.image(str(config.TREE_PNG))


# ---------------------------------------------------------------------------
# Page 3: About
# ---------------------------------------------------------------------------
def show_about_page():
    st.title("About and Responsible Use")
    st.markdown(
        """
**What the system does**
- Trains a Decision Tree classifier on labelled synthetic examples.
- Predicts one of three categories: Needs Support, Meeting Expectations, Exceeding Expectations.
- Shows the questions the tree asked and gives rule-based improvement suggestions.

**What it does NOT do**
- It does not use real employee data, personal attributes, or surveillance.
- It does not decide hiring, firing, promotion, pay or penalties.
- Its labels come from a formula plus noise, so the model mostly learns that formula.

**Responsible use**
- Real data could be biased or incomplete, and the model would inherit those problems.
- A prediction is a starting point for a human conversation, never a verdict.
"""
    )


# ---------------------------------------------------------------------------
# Main: choose the page
# ---------------------------------------------------------------------------
def main():
    page = st.sidebar.radio("Page", ["Predict", "Model Evaluation", "About"])
    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    if page == "Predict":
        show_predict_page(bundle)
    elif page == "Model Evaluation":
        show_evaluation_page()
    else:
        show_about_page()


main()
