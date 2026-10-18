"""
generate_dataset.py - creates a SYNTHETIC (fictional) employee dataset.

IMPORTANT HONESTY NOTE
----------------------
These are NOT real employee records. The numbers are produced by a random
generator, and the performance category is produced by a formula plus random
noise. A model trained on this data mostly learns OUR formula, not real-world
employee behaviour. Good accuracy here shows that the machine-learning
pipeline works; it does NOT prove the model would work in a real company.

How the data is built (assumptions)
-----------------------------------
1. A hidden "diligence" number is drawn for each fictional employee. It makes
   the features partly correlated (an employee who completes more tasks also
   tends to deliver on time and make fewer errors), as in real life.
2. Each feature = a base value + effect of diligence + random noise,
   then clipped to its valid range.
3. A hidden weighted "composite score" is computed from the features:
       task completion 30%, on-time delivery 25%, error rate 20%,
       attendance 10%, peer collaboration 10%, training hours 5%.
   Random noise is added, so two similar-looking employees can end up in
   different classes.
4. The lowest 25% of scores -> "Needs Support", the middle 50% ->
   "Meeting Expectations", the highest 25% -> "Exceeding Expectations".
5. About 5% of labels are then moved to a neighbouring class to imitate
   disagreement between managers ("label noise").

Run:  python generate_dataset.py
"""

import numpy as np
import pandas as pd

import config


def generate_employee_data(n_records: int, seed: int) -> pd.DataFrame:
    """Return a DataFrame of fictional employees with a performance category."""
    rng = np.random.default_rng(seed)  # reproducible random numbers

    # Step 1: hidden diligence value (mean 0, spread 1) for each employee
    diligence = rng.normal(0, 1, n_records)

    # Step 2: build each feature, then clip to its valid range
    task_completion = np.clip(80 + 8 * diligence + rng.normal(0, 6, n_records), 40, 100)
    on_time = np.clip(78 + 9 * diligence + rng.normal(0, 8, n_records), 30, 100)
    error_rate = np.clip(6 - 2 * diligence + rng.normal(0, 2.5, n_records), 0, 20)
    attendance = np.clip(93 + 3 * diligence + rng.normal(0, 3.5, n_records), 60, 100)
    training = np.clip(15 + 4 * diligence + rng.normal(0, 8, n_records), 0, 40)
    peer_score = np.clip(3.5 + 0.4 * diligence + rng.normal(0, 0.6, n_records), 1, 5)

    data = pd.DataFrame({
        config.ID_COLUMN: [f"E{i + 1:03d}" for i in range(n_records)],
        "task_completion_rate": task_completion.round(1),
        "on_time_delivery_rate": on_time.round(1),
        "error_rate": error_rate.round(1),
        "attendance_pct": attendance.round(1),
        "training_hours": training.round(1),
        "peer_collaboration_score": peer_score.round(1),
    })

    # Step 3: hidden composite score (each part scaled to roughly 0-1)
    score = (
        0.30 * (data["task_completion_rate"] - 40) / 60
        + 0.25 * (data["on_time_delivery_rate"] - 30) / 70
        + 0.20 * (1 - data["error_rate"] / 20)
        + 0.10 * (data["attendance_pct"] - 60) / 40
        + 0.05 * (data["training_hours"] / 40)
        + 0.10 * (data["peer_collaboration_score"] - 1) / 4
    )
    score = score + rng.normal(0, 0.04, n_records)  # noise: similar people can differ

    # Step 4: cut the score into 3 classes (25% / 50% / 25%)
    low_cut, high_cut = np.quantile(score, [0.25, 0.75])
    class_index = np.where(score < low_cut, 0, np.where(score < high_cut, 1, 2))

    # Step 5: label noise - move about 5% of labels to a neighbouring class
    n_flip = int(0.05 * n_records)
    flip_positions = rng.choice(n_records, size=n_flip, replace=False)
    for position in flip_positions:
        if class_index[position] == 1:
            class_index[position] = rng.choice([0, 2])   # middle -> low or high
        else:
            class_index[position] = 1                    # extreme -> middle

    data[config.TARGET_COLUMN] = [config.CLASS_LABELS[i] for i in class_index]
    return data


def main() -> None:
    data = generate_employee_data(config.N_RECORDS, config.RANDOM_SEED)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)   # create data/ if missing
    data.to_csv(config.DATA_PATH, index=False)

    print(f"Saved {len(data)} synthetic records to: {config.DATA_PATH}")
    print("\nClass distribution:")
    print(data[config.TARGET_COLUMN].value_counts().reindex(config.CLASS_LABELS))
    print("\nFirst 5 rows:")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()
