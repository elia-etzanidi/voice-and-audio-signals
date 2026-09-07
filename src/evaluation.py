import json

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def time_to_seconds(time_str):
    parts = time_str.strip().split(':')

    return (
        float(parts[0]) * 3600
        + float(parts[1]) * 60
        + float(parts[2])
    )


def load_ground_truth_intervals(json_path, target_session="S01"):
    """
    Loads all speech intervals for the target session from a JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    speech_intervals = []

    for entry in data:
        if entry.get("session_id") == target_session:
            start_sec = time_to_seconds(entry["start_time"])
            end_sec = time_to_seconds(entry["end_time"])

            speech_intervals.append((start_sec, end_sec))

    return speech_intervals


def create_frame_ground_truth(
    speech_intervals,
    total_frames,
    hop_length,
    sample_rate
):
    y_true = np.zeros(total_frames, dtype=int)

    frame_times = np.arange(total_frames) * (
        hop_length / sample_rate
    )

    for start_sec, end_sec in speech_intervals:
        mask = (
            (frame_times >= start_sec)
            & (frame_times <= end_sec)
        )

        y_true[mask] = 1

    return y_true


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """
    Evaluates predictions using the classification report,
    confusion matrix, and macro F1-score.
    """
    print(
        f"\n================ Evaluation: "
        f"{model_name} ================"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Background", "Foreground"],
            digits=4
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    return f1_score(
        y_true,
        y_pred,
        average="macro"
    )