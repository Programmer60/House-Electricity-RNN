"""TensorFlow/Keras vanilla RNN baseline for household power forecasting."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .evaluate_numpy import calculate_metrics, load_target_scaling
    from .train_numpy import load_rds_sequences
except ImportError:
    from evaluate_numpy import calculate_metrics, load_target_scaling
    from train_numpy import load_rds_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_predictions_csv(actual_kw: np.ndarray, predicted_kw: np.ndarray, output_path: Path) -> None:
    """Save actual, predicted, and signed error values in kW."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_kW", "predicted_kW", "error_kW"])
        writer.writerows(
            (actual, predicted, actual - predicted)
            for actual, predicted in zip(actual_kw, predicted_kw)
        )


def save_loss_plot(history: dict[str, list[float]], output_path: Path) -> None:
    """Save training and validation normalized MSE histories."""
    epochs = np.arange(1, len(history["training_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["training_loss"], marker="o", label="Training loss")
    axis.plot(epochs, history["validation_loss"], marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss (scaled target)")
    axis.set_title("TensorFlow/Keras vanilla RNN training and validation loss")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_prediction_plot(
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    output_path: Path,
    representative_count: int = 500,
) -> None:
    """Save representative test predictions and an actual/predicted scatter plot."""
    count = min(representative_count, actual_kw.size)
    lower = float(min(actual_kw.min(), predicted_kw.min()))
    upper = float(max(actual_kw.max(), predicted_kw.max()))
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    samples = np.arange(count)
    axes[0].plot(samples, actual_kw[:count], label="Actual", linewidth=1.2)
    axes[0].plot(samples, predicted_kw[:count], label="Predicted", linewidth=1.0)
    axes[0].set_title(f"First {count} unseen test samples")
    axes[0].set_xlabel("Test sample")
    axes[0].set_ylabel("Global_active_power (kW)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].scatter(actual_kw, predicted_kw, s=8, alpha=0.25)
    axes[1].plot([lower, upper], [lower, upper], "r--", label="Ideal y = x")
    axes[1].set_title("Actual versus predicted test power")
    axes[1].set_xlabel("Actual Global_active_power (kW)")
    axes[1].set_ylabel("Predicted Global_active_power (kW)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_error_plot(actual_kw: np.ndarray, predicted_kw: np.ndarray, output_path: Path) -> None:
    """Save signed test errors in kW against test sample."""
    errors = actual_kw - predicted_kw
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(np.arange(errors.size), errors, linewidth=0.7)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("TensorFlow/Keras vanilla RNN prediction error")
    axis.set_xlabel("Test sample")
    axis.set_ylabel("Error (actual - predicted, kW)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def load_comparison(results_dir: Path) -> dict[str, dict[str, float | int | str]]:
    """Load existing baseline metrics without retraining those models."""
    comparison: dict[str, dict[str, float | int | str]] = {}
    numpy_metrics = json.loads(
        (results_dir / "rnn_test_metrics.json").read_text(encoding="utf-8")
    )["metrics"]
    comparison["NumPy RNN"] = {"optimizer": "SGD", **numpy_metrics}
    pytorch_path = results_dir / "rnn_pytorch_metrics.json"
    if pytorch_path.exists():
        pytorch = json.loads(pytorch_path.read_text(encoding="utf-8"))
        comparison["PyTorch RNN"] = {
            "epochs": pytorch["best_epoch"],
            "optimizer": "Adam",
            **pytorch["test_metrics"],
        }
    return comparison


def save_comparison(
    comparison: dict[str, dict[str, float | int | str]],
    output_path: Path,
) -> None:
    """Save metrics for the NumPy, PyTorch, and Keras baselines."""
    lines = [
        "# NumPy, PyTorch, and TensorFlow/Keras Vanilla RNN Comparison",
        "",
        "| Model | Epochs | Optimizer | Test MAE (kW) | Test RMSE (kW) | Test R^2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, values in comparison.items():
        lines.append(
            f"| {name} | {values.get('epochs', 'n/a')} | {values['optimizer']} | "
            f"{values['MAE_kW']:.8f} | {values['RMSE_kW']:.8f} | {values['R2']:.8f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_and_evaluate(
    results_dir: Path,
    models_dir: Path,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    patience: int = 5,
    validation_fraction: float = 0.10,
    seed: int = 42,
) -> dict[str, object]:
    """Train a Keras SimpleRNN and evaluate the untouched chronological test set."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is required. Install it with `python -m pip install tensorflow`."
        ) from error

    tf.keras.utils.set_random_seed(seed)
    train_path = PROJECT_ROOT / "preprocessing" / "train_sequences.rds"
    test_path = PROJECT_ROOT / "preprocessing" / "test_sequences.rds"
    metadata_path = PROJECT_ROOT / "preprocessing" / "preprocessing_metadata.rds"
    X_train_all, y_train_all = load_rds_sequences(train_path)
    X_test, y_test_scaled = load_rds_sequences(test_path)
    if X_train_all.shape != (27647, 24, 7) or y_train_all.shape != (27647,):
        raise ValueError(f"Unexpected train shapes: {X_train_all.shape}, {y_train_all.shape}")
    if X_test.shape != (6894, 24, 7) or y_test_scaled.shape != (6894,):
        raise ValueError(f"Unexpected test shapes: {X_test.shape}, {y_test_scaled.shape}")

    validation_samples = int(round(X_train_all.shape[0] * validation_fraction))
    split_index = X_train_all.shape[0] - validation_samples
    X_train, y_train = X_train_all[:split_index], y_train_all[:split_index]
    X_validation, y_validation = X_train_all[split_index:], y_train_all[split_index:]
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(24, 7)),
            tf.keras.layers.SimpleRNN(32, activation="tanh"),
            tf.keras.layers.Dense(1),
        ],
        name="tensorflow_vanilla_rnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
        restore_best_weights=True,
    )
    start_time = time.perf_counter()
    fit_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=max_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        verbose=1,
    )
    runtime_seconds = time.perf_counter() - start_time
    history = {
        "training_loss": [float(value) for value in fit_history.history["loss"]],
        "validation_loss": [float(value) for value in fit_history.history["val_loss"]],
    }
    best_epoch = int(np.argmin(history["validation_loss"])) + 1
    best_validation_loss = history["validation_loss"][best_epoch - 1]
    predictions_scaled = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
    target_min, target_max = load_target_scaling(metadata_path)
    scale = target_max - target_min
    predictions_kw = predictions_scaled * scale + target_min
    actual_kw = y_test_scaled * scale + target_min
    if not np.isfinite(predictions_kw).all():
        raise FloatingPointError("NaN or Inf found in TensorFlow/Keras predictions")
    metrics = calculate_metrics(actual_kw, predictions_kw)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "rnn_tensorflow_best.keras"
    model.save(model_path)
    predictions_path = results_dir / "rnn_tensorflow_predictions.csv"
    loss_plot_path = results_dir / "rnn_tensorflow_loss.png"
    actual_plot_path = results_dir / "rnn_tensorflow_actual_vs_predicted.png"
    error_plot_path = results_dir / "rnn_tensorflow_error.png"
    save_predictions_csv(actual_kw, predictions_kw, predictions_path)
    save_loss_plot(history, loss_plot_path)
    save_prediction_plot(actual_kw, predictions_kw, actual_plot_path)
    save_error_plot(actual_kw, predictions_kw, error_plot_path)
    comparison = load_comparison(results_dir)
    comparison["TensorFlow/Keras RNN"] = {
        "epochs": best_epoch,
        "optimizer": "Adam",
        **metrics,
    }
    comparison_path = results_dir / "rnn_tensorflow_comparison.md"
    save_comparison(comparison, comparison_path)
    result = {
        "model": "TensorFlow/Keras Vanilla RNN",
        "configuration": {
            "input_size": 7,
            "hidden_units": 32,
            "timesteps": 24,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "max_epochs": max_epochs,
            "patience": patience,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "shuffle": False,
        },
        "split": {
            "train_samples": int(X_train.shape[0]),
            "validation_samples": int(X_validation.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "split_index": int(split_index),
        },
        "epochs_trained": len(history["training_loss"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history,
        "test_metrics": metrics,
        "prediction_distribution": {
            "minimum_kW": float(predictions_kw.min()),
            "maximum_kW": float(predictions_kw.max()),
            "mean_kW": float(predictions_kw.mean()),
        },
        "runtime_seconds": runtime_seconds,
        "comparison": comparison,
        "artifacts": {
            "model": str(model_path),
            "metrics": str(results_dir / "rnn_tensorflow_metrics.json"),
            "config": str(results_dir / "rnn_tensorflow_config.json"),
            "predictions": str(predictions_path),
            "loss_plot": str(loss_plot_path),
            "actual_vs_predicted_plot": str(actual_plot_path),
            "error_plot": str(error_plot_path),
            "comparison": str(comparison_path),
        },
    }
    (results_dir / "rnn_tensorflow_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (results_dir / "rnn_tensorflow_config.json").write_text(
        json.dumps(result["configuration"], indent=2), encoding="utf-8"
    )
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_validation_loss)
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")
    print("Saved TensorFlow/Keras artifacts in:", results_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    train_and_evaluate(
        args.results_dir,
        args.models_dir,
        args.max_epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
        args.validation_fraction,
        args.seed,
    )


if __name__ == "__main__":
    main()
