from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time


def train_knn(X_train, y_train, n_neighbors=5):
    """
    Trains a k-NN classifier.
    """
    print(f"\n[k-NN] Training with k={n_neighbors}...")

    start_time = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='uniform',
        n_jobs=-1
    )

    knn.fit(X_train, y_train)

    elapsed = time.time() - start_time

    print(f"[k-NN] Completed in {elapsed:.2f}s.")

    return knn


def train_mlp(X_train, y_train, hidden_layer_sizes=(64, 32), max_iter=200):
    """
    Trains a Multi-Layer Perceptron with two hidden layers.
    """
    print(
        f"\n[MLP] Training 2-Layer MLP "
        f"{hidden_layer_sizes}..."
    )

    start_time = time.time()

    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=max_iter,
        early_stopping=True,
        random_state=42,
        verbose=False
    )

    mlp.fit(X_train, y_train)

    elapsed = time.time() - start_time

    print(
        f"[MLP] Completed in {elapsed:.2f}s "
        f"(Epochs: {mlp.n_iter_})."
    )

    return mlp