import csv
import numpy as np
from scipy.ndimage import median_filter


def smooth_predictions(predictions, window_size=51):
    """
    Applies a median filter to smooth binary predictions and
    reduce short-term classification errors.

    With a hop length of 10 ms, 51 frames correspond to
    approximately 0.5 seconds.
    """
    # The filter window must have an odd size.
    if window_size % 2 == 0:
        window_size += 1

    smoothed = median_filter(predictions, size=window_size)

    return smoothed


def get_segments(predictions, hop_length_sec):
    """
    Converts a sequence of binary predictions into
    time segments of the form (start, end, class_name).
    """
    segments = []

    current_class = predictions[0]
    start_frame = 0

    for i in range(1, len(predictions)):
        if predictions[i] != current_class:
            end_frame = i

            start_time = start_frame * hop_length_sec
            end_time = end_frame * hop_length_sec

            class_name = (
                "foreground" if current_class == 1 else "background"
            )

            segments.append(
                (start_time, end_time, class_name)
            )

            start_frame = i
            current_class = predictions[i]

    # Add the final segment.
    end_time = len(predictions) * hop_length_sec
    start_time = start_frame * hop_length_sec

    class_name = (
        "foreground" if current_class == 1 else "background"
    )

    segments.append(
        (start_time, end_time, class_name)
    )

    return segments


def export_segments_to_csv(segments, output_path, audio_filename):
    """
    Exports the detected segments to a CSV file.
    """
    with open(
        output_path,
        mode='w',
        newline='',
        encoding='utf-8'
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["Audiofile", "start", "end", "class"]
        )

        for start, end, label in segments:
            writer.writerow([
                audio_filename,
                f"{start:.3f}",
                f"{end:.3f}",
                label
            ])

    print(f"File successfully exported to: {output_path}")