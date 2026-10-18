"""
recommender.py - simple, rule-based improvement suggestions.

THIS IS NOT MACHINE LEARNING. It is a set of if-statements that compare the
entered values with fixed thresholds. The thresholds below are illustrative
values chosen by us for teaching; they are not learned from data and are not
official HR standards. The ML model's prediction is NOT used here.
"""

# Illustrative thresholds (change them if your faculty suggests other values)
LOW_TASK_COMPLETION = 75       # below this -> suggest smaller milestones
LOW_ON_TIME = 70               # below this -> suggest deadline planning
HIGH_ERROR_RATE = 8            # above this -> suggest checklists / peer review
LOW_ATTENDANCE = 90            # below this -> suggest a supportive conversation
LOW_TRAINING_HOURS = 8         # below this -> suggest skill-development activities
LOW_PEER_SCORE = 3.0           # below this -> suggest collaboration practices

# Values that count as a visible strength
STRONG_TASK_COMPLETION = 90
STRONG_ON_TIME = 90
STRONG_ERROR_RATE = 3
STRONG_ATTENDANCE = 97
STRONG_TRAINING_HOURS = 25
STRONG_PEER_SCORE = 4.2


def get_recommendations(values: dict) -> dict:
    """
    Return {"improvements": [...], "strengths": [...]}.
    Each improvement is a dict with 'indicator', 'observation' and 'suggestion'.
    Expects values that already passed predictor.validate_inputs().
    """
    improvements = []
    strengths = []

    completion = float(values["task_completion_rate"])
    on_time = float(values["on_time_delivery_rate"])
    errors = float(values["error_rate"])
    attendance = float(values["attendance_pct"])
    training = float(values["training_hours"])
    peer = float(values["peer_collaboration_score"])

    # ---- improvement areas -------------------------------------------------
    if completion < LOW_TASK_COMPLETION:
        improvements.append({
            "indicator": "Task completion",
            "observation": f"Completion rate is {completion:g}% (below {LOW_TASK_COMPLETION}%).",
            "suggestion": "Break large tasks into smaller milestones and review progress weekly "
                          "with a supervisor.",
        })
    if on_time < LOW_ON_TIME:
        improvements.append({
            "indicator": "Timeliness",
            "observation": f"On-time delivery is {on_time:g}% (below {LOW_ON_TIME}%).",
            "suggestion": "Plan the workload at the start of each week, set deadline reminders, "
                          "and raise a request early when a deadline looks unrealistic.",
        })
    if errors > HIGH_ERROR_RATE:
        improvements.append({
            "indicator": "Quality",
            "observation": f"Error rate is {errors:g} per 100 tasks (above {HIGH_ERROR_RATE}).",
            "suggestion": "Use a self-check list before submitting work and ask a colleague "
                          "for a quick peer review on critical tasks.",
        })
    if attendance < LOW_ATTENDANCE:
        improvements.append({
            "indicator": "Attendance",
            "observation": f"Attendance is {attendance:g}% (below {LOW_ATTENDANCE}%).",
            "suggestion": "Have a supportive, private conversation to understand any "
                          "obstacles; do not assume the cause.",
        })
    if training < LOW_TRAINING_HOURS:
        improvements.append({
            "indicator": "Skill development",
            "observation": f"Only {training:g} training hours last quarter (below {LOW_TRAINING_HOURS}).",
            "suggestion": "Pick one relevant short course or workshop for next quarter and "
                          "schedule protected time for it.",
        })
    if peer < LOW_PEER_SCORE:
        improvements.append({
            "indicator": "Collaboration",
            "observation": f"Peer collaboration score is {peer:g} (below {LOW_PEER_SCORE}).",
            "suggestion": "Join regular team check-ins and ask teammates what kind of "
                          "support would help them most.",
        })

    # ---- visible strengths -------------------------------------------------
    if completion >= STRONG_TASK_COMPLETION:
        strengths.append(f"High task completion ({completion:g}%).")
    if on_time >= STRONG_ON_TIME:
        strengths.append(f"Reliable on-time delivery ({on_time:g}%).")
    if errors <= STRONG_ERROR_RATE:
        strengths.append(f"Low error rate ({errors:g} per 100 tasks).")
    if attendance >= STRONG_ATTENDANCE:
        strengths.append(f"Excellent attendance ({attendance:g}%).")
    if training >= STRONG_TRAINING_HOURS:
        strengths.append(f"Strong investment in skill development ({training:g} hours).")
    if peer >= STRONG_PEER_SCORE:
        strengths.append(f"Highly rated by peers ({peer:g}/5).")

    return {"improvements": improvements, "strengths": strengths}
