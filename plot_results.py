import librosa
import matplotlib.pyplot as plt
import csv
import numpy as np

from src.evaluation import load_ground_truth_intervals
from src.config import TEST_AUDIO, TEST_JSON, KNN_CSV, MLP_CSV, PLOT_LIMIT_SEC


def load_predictions_from_csv(csv_path):
    """
    Loads model predictions from a CSV file and returns
    the intervals classified as foreground (speech).
    """
    intervals = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)

        for row in reader:
            start = float(row[1])
            end = float(row[2])
            label = row[3]

            if label == "foreground":
                intervals.append((start, end))

    return intervals


def plot_intervals(ax, intervals, color, limit_sec):
    """Plots speech intervals as shaded regions."""
    for start, end in intervals:
        if start > limit_sec:
            break

        actual_end = min(end, limit_sec)
        ax.axvspan(start, actual_end, color=color, alpha=0.4, lw=0)


def main():
    # Load only the first 120 seconds to limit memory usage.
    y, sr = librosa.load(
        TEST_AUDIO,
        sr=16000,
        duration=PLOT_LIMIT_SEC
    )

    time_axis = np.linspace(0, PLOT_LIMIT_SEC, num=len(y))

    gt_intervals = load_ground_truth_intervals(
        TEST_JSON,
        target_session="S01"
    )

    knn_intervals = load_predictions_from_csv(KNN_CSV)
    mlp_intervals = load_predictions_from_csv(MLP_CSV)

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 9),
        sharex=True
    )

    fig.suptitle(
        "Speech Segmentation: Ground Truth vs Models (First 120s)",
        fontsize=16
    )

    titles = [
        "Ground Truth (S01.json)",
        "k-NN Predictions",
        "2-Layer MLP Predictions"
    ]

    interval_lists = [
        gt_intervals,
        knn_intervals,
        mlp_intervals
    ]

    colors = ['green', 'blue', 'orange']

    for i, ax in enumerate(axes):
        ax.plot(
            time_axis,
            y,
            color='gray',
            alpha=0.5,
            rasterized=True
        )

        plot_intervals(
            ax,
            interval_lists[i],
            colors[i],
            PLOT_LIMIT_SEC
        )

        ax.plot(
            [],
            [],
            color=colors[i],
            alpha=0.4,
            label=titles[i]
        )

        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel("Time (seconds)")

    plt.tight_layout()

    plot_path = "results/segmentation_plot.png"
    plt.savefig(plot_path, dpi=300)

    print(f"\nGraph saved successfully to: {plot_path}")


if __name__ == "__main__":
    main()