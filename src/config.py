# --- Audio Processing ---
SAMPLE_RATE = 16000
FRAME_LENGTH = int(0.025 * SAMPLE_RATE)  # 25 ms
HOP_LENGTH = int(0.010 * SAMPLE_RATE)    # 10 ms
N_MFCC = 13

# --- File Paths ---
TRAIN_DIR = "data/train"
TEST_AUDIO = "data/test/S01_U04.CH4.wav"
TEST_JSON = "data/test/S01.json"
RESULTS_DIR = "results"

# --- Results ---
KNN_CSV = "results/predictions_knn.csv"
MLP_CSV = "results/predictions_mlp.csv"

# --- Plotting ---
PLOT_LIMIT_SEC = 120