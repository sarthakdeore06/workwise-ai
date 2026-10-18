"""
config.py - shared settings for WorkWise AI.

Every other file imports its feature names, class labels, value ranges and
file paths from here. Keeping them in ONE place guarantees that the training
code (train_model.py) and the prediction code (predictor.py / app.py) always
use exactly the same columns in exactly the same order.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File locations (built from this file's own folder, so they work on any PC)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "employee_performance.csv"            # made by generate_dataset.py
MODEL_PATH = MODEL_DIR / "workwise_model.joblib"             # made by train_model.py
METRICS_PATH = MODEL_DIR / "metrics.json"                    # made by train_model.py
CONFUSION_MATRIX_PNG = MODEL_DIR / "confusion_matrix.png"    # made by train_model.py
TREE_PNG = MODEL_DIR / "decision_tree.png"                   # made by train_model.py
MISCLASSIFIED_PATH = MODEL_DIR / "misclassified_test_records.csv"  # made by train_model.py
TEST_RESULTS_PATH = MODEL_DIR / "test_results.csv"           # made by test_cases.py

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42      # fixed seed -> same dataset and same split every run
N_RECORDS = 600       # number of fictional employees to generate
TEST_SIZE = 0.20      # 20% of records are held out for testing

# ---------------------------------------------------------------------------
# Columns and labels
# ---------------------------------------------------------------------------
ID_COLUMN = "employee_id"            # identifier only - NEVER used as a feature
TARGET_COLUMN = "performance_category"

# The six input features, in the exact order the model expects them.
FEATURE_COLUMNS = [
    "task_completion_rate",
    "on_time_delivery_rate",
    "error_rate",
    "attendance_pct",
    "training_hours",
    "peer_collaboration_score",
]

# The three target classes (lowest to highest performance).
CLASS_LABELS = [
    "Needs Support",
    "Meeting Expectations",
    "Exceeding Expectations",
]

# Description, valid range and default for each feature (used for validation
# in predictor.py and for the input boxes in app.py).
FEATURE_INFO = {
    "task_completion_rate": {
        "label": "Task completion rate (%)",
        "min": 40.0, "max": 100.0, "default": 80.0, "step": 1.0,
        "help": "Percentage of assigned tasks completed in the review period.",
    },
    "on_time_delivery_rate": {
        "label": "On-time delivery rate (%)",
        "min": 30.0, "max": 100.0, "default": 78.0, "step": 1.0,
        "help": "Percentage of completed tasks delivered on or before the deadline.",
    },
    "error_rate": {
        "label": "Error rate (mistakes per 100 tasks)",
        "min": 0.0, "max": 20.0, "default": 6.0, "step": 0.5,
        "help": "Number of reviewer-reported mistakes per 100 tasks. Lower is better.",
    },
    "attendance_pct": {
        "label": "Attendance (%)",
        "min": 60.0, "max": 100.0, "default": 93.0, "step": 1.0,
        "help": "Percentage of scheduled working days attended (approved leave excluded).",
    },
    "training_hours": {
        "label": "Training hours (last quarter)",
        "min": 0.0, "max": 40.0, "default": 15.0, "step": 1.0,
        "help": "Hours spent on skill-development courses or workshops last quarter.",
    },
    "peer_collaboration_score": {
        "label": "Peer collaboration score (1-5)",
        "min": 1.0, "max": 5.0, "default": 3.5, "step": 0.1,
        "help": "Average rating given by teammates on helpfulness and teamwork (1 = low, 5 = high).",
    },
}

# ---------------------------------------------------------------------------
# Sample fictional employees used by the app's drop-down and by test_cases.py
# ---------------------------------------------------------------------------
SAMPLE_EMPLOYEES = {
    "Typical employee": {
        "task_completion_rate": 80.0, "on_time_delivery_rate": 78.0,
        "error_rate": 6.0, "attendance_pct": 93.0,
        "training_hours": 15.0, "peer_collaboration_score": 3.5,
    },
    "Struggling employee": {
        "task_completion_rate": 55.0, "on_time_delivery_rate": 50.0,
        "error_rate": 14.0, "attendance_pct": 82.0,
        "training_hours": 3.0, "peer_collaboration_score": 2.2,
    },
    "High performer": {
        "task_completion_rate": 96.0, "on_time_delivery_rate": 95.0,
        "error_rate": 1.5, "attendance_pct": 98.0,
        "training_hours": 30.0, "peer_collaboration_score": 4.6,
    },
    "Fast but often late": {
        "task_completion_rate": 92.0, "on_time_delivery_rate": 55.0,
        "error_rate": 5.0, "attendance_pct": 94.0,
        "training_hours": 14.0, "peer_collaboration_score": 3.6,
    },
}
