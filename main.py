import os

from sklearn.preprocessing import StandardScaler

from src.config import (
    TRAIN_DIR, 
    SAMPLE_RATE, 
    HOP_LENGTH, 
    RESULTS_DIR,
    TEST_JSON,
    TEST_AUDIO
)
from src.dataset import load_dataset
from src.audio_features import extract_features_from_audio
from src.models import train_knn, train_mlp
from src.postprocessing import (
    smooth_predictions,
    get_segments,
    export_segments_to_csv
)
from src.evaluation import (
    load_ground_truth_intervals,
    create_frame_ground_truth,
    evaluate_predictions
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print(" 1. Loading and Preparing Training Data")
    print("=" * 60)

    # Limit the number of files per class for balanced and efficient training.
    X_train, y_train = load_dataset(
        TRAIN_DIR,
        max_files_per_class=30
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"X_train_scaled shape: {X_train_scaled.shape}")

    print("\n" + "=" * 60)
    print(" 2. Training Classifiers (k-NN & MLP)")
    print("=" * 60)

    knn_model = train_knn(
        X_train_scaled,
        y_train,
        n_neighbors=5
    )

    mlp_model = train_mlp(
        X_train_scaled,
        y_train,
        hidden_layer_sizes=(64, 32)
    )

    print("\n" + "=" * 60)
    print(" 3. Extracting Test Audio Features")
    print("=" * 60)

    X_test = extract_features_from_audio(TEST_AUDIO)

    # Use the training scaler without refitting it on the test data.
    X_test_scaled = scaler.transform(X_test)

    print(f"X_test_scaled shape: {X_test_scaled.shape}")

    print("\n" + "=" * 60)
    print(" 4. Predictions and Post-processing")
    print("=" * 60)

    y_pred_knn_raw = knn_model.predict(X_test_scaled)
    y_pred_mlp_raw = mlp_model.predict(X_test_scaled)

    # Apply a median filter over approximately 0.5 seconds.
    window_size = 51

    y_pred_knn_smooth = smooth_predictions(
        y_pred_knn_raw,
        window_size=window_size
    )

    y_pred_mlp_smooth = smooth_predictions(
        y_pred_mlp_raw,
        window_size=window_size
    )

    print("\n" + "=" * 60)
    print(" 5. Evaluation Against Ground Truth (S01.json - U04)")
    print("=" * 60)

    hop_length_sec = HOP_LENGTH / SAMPLE_RATE
    total_test_frames = X_test_scaled.shape[0]

    intervals = load_ground_truth_intervals(
        TEST_JSON,
        target_session="S01"
    )

    y_true = create_frame_ground_truth(
        intervals,
        total_test_frames,
        HOP_LENGTH,
        SAMPLE_RATE
    )

    evaluate_predictions(
        y_true,
        y_pred_knn_smooth,
        model_name="k-NN (Smoothed)"
    )

    evaluate_predictions(
        y_true,
        y_pred_mlp_smooth,
        model_name="2-Layer MLP (Smoothed)"
    )

    print("\n" + "=" * 60)
    print(" 6. Exporting Results to CSV")
    print("=" * 60)

    audio_filename = os.path.basename(TEST_AUDIO)

    segments_knn = get_segments(
        y_pred_knn_smooth,
        hop_length_sec
    )

    csv_path_knn = os.path.join(
        RESULTS_DIR,
        "predictions_knn.csv"
    )

    export_segments_to_csv(
        segments_knn,
        csv_path_knn,
        audio_filename
    )

    segments_mlp = get_segments(
        y_pred_mlp_smooth,
        hop_length_sec
    )

    csv_path_mlp = os.path.join(
        RESULTS_DIR,
        "predictions_mlp.csv"
    )

    export_segments_to_csv(
        segments_mlp,
        csv_path_mlp,
        audio_filename
    )

    print(
        f"\nPipeline completed successfully! "
        f"CSV files are located in '{RESULTS_DIR}'."
    )


if __name__ == "__main__":
    main()