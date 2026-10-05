"""Evaluate the saved NumPy vanilla RNN on the unseen test sequences."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .rnn_numpy import VanillaRNN
    from .train_numpy import load_rds_sequences
except ImportError:
    from rnn_numpy import VanillaRNN
    from train_numpy import load_rds_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PARAMETERS = ("W_xh", "W_hh", "b_h", "W_hy", "b_y")


def find_rscript(explicit_path: str | None = None) -> str:
    """Find the installed Rscript used to read the existing RDS metadata."""
    candidates = [explicit_path, os.environ.get("RSCRIPT"), shutil.which("Rscript")]
    candidates.extend(str(path) for path in Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Rscript was not found. Pass --rscript with the R installation path."
    )


def load_target_scaling(metadata_path: Path, rscript_path: str | None = None) -> tuple[float, float]:
    """Read Global_active_power min/max from preprocessing metadata."""
    rscript = find_rscript(rscript_path)
    expression = (
        "args <- commandArgs(trailingOnly=TRUE); "
        "metadata <- readRDS(args[1]); "
        "cat(metadata$mins['Global_active_power'], metadata$maxs['Global_active_power'])"
    )
    completed = subprocess.run(
        [rscript, "-e", expression, str(metadata_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [float(value) for value in completed.stdout.split()]
    if len(values) != 2 or values[1] <= values[0]:
        raise ValueError(f"Invalid target scaling values: {values}")
    return values[0], values[1]


def load_saved_model(model_path: Path) -> VanillaRNN:
    """Load and validate all saved baseline parameters."""
    saved = np.load(model_path)
    missing = [name for name in REQUIRED_PARAMETERS if name not in saved]
    if missing:
        raise ValueError(f"Saved model is missing parameters: {missing}")
    if saved["W_xh"].shape != (32, 7):
        raise ValueError(f"Expected W_xh shape (32, 7), got {saved['W_xh'].shape}")
    if saved["W_hh"].shape != (32, 32):
        raise ValueError(f"Expected W_hh shape (32, 32), got {saved['W_hh'].shape}")
    if saved["b_h"].shape != (32,) or saved["W_hy"].shape != (1, 32) or saved["b_y"].shape != (1,):
        raise ValueError("Saved model parameter shapes do not match the 32-unit baseline")

    model = VanillaRNN(input_size=7, hidden_size=32, output_size=1, seed=42)
    for name in REQUIRED_PARAMETERS:
        setattr(model, name, saved[name].copy())
    return model


def generate_predictions(model: VanillaRNN, X_test: np.ndarray) -> np.ndarray:
    """Run forward propagation only; no parameters are updated."""
    predictions = np.empty(X_test.shape[0], dtype=float)
    for index, sequence in enumerate(X_test):
        predictions[index] = model.forward(sequence)[0][0]
    return predictions


def calculate_metrics(actual_kw: np.ndarray, predicted_kw: np.ndarray) -> dict[str, float]:
    """Calculate metrics on the original kW scale."""
    errors = actual_kw - predicted_kw
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((actual_kw - np.mean(actual_kw)) ** 2))
    r_squared = float(1.0 - np.sum(errors**2) / denominator)
    return {"MAE_kW": mae, "MSE_kW2": mse, "RMSE_kW": rmse, "R2": r_squared}


def save_evaluation_plot(
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    output_path: Path,
    representative_count: int = 500,
) -> None:
    """Save representative time-series and actual-vs-predicted scatter plots."""
    count = min(representative_count, actual_kw.size)
    samples = np.arange(count)
    lower = float(min(actual_kw.min(), predicted_kw.min()))
    upper = float(max(actual_kw.max(), predicted_kw.max()))

    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(samples, actual_kw[:count], label="Actual", linewidth=1.2)
    axes[0].plot(samples, predicted_kw[:count], label="Predicted", linewidth=1.0)
    axes[0].set_title(f"Global active power: first {count} test samples")
    axes[0].set_xlabel("Test sample")
    axes[0].set_ylabel("Global_active_power (kW)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].scatter(actual_kw, predicted_kw, s=8, alpha=0.25)
    axes[1].plot([lower, upper], [lower, upper], "r--", label="Ideal y = x")
    axes[1].set_title("Actual versus predicted test power")
    axes[1].set_xlabel("Actual Global_active_power (kW)")
    axes[1].set_ylabel("Predicted Global_active_power (kW)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_error_plot(actual_kw: np.ndarray, predicted_kw: np.ndarray, output_path: Path) -> None:
    """Save signed error e_i = y_i - y_hat_i against test sample."""
    errors = actual_kw - predicted_kw
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(np.arange(errors.size), errors, linewidth=0.7)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Prediction error on unseen test samples")
    axis.set_xlabel("Test sample")
    axis.set_ylabel("Error (actual - predicted, kW)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_predictions_csv(
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    output_path: Path,
) -> None:
    """Write actual, predicted, and signed error values in kW."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_kW", "predicted_kW", "error_kW"])
        writer.writerows(
            (actual, predicted, actual - predicted)
            for actual, predicted in zip(actual_kw, predicted_kw)
        )


def save_report(
    metrics: dict[str, float],
    prediction_range: tuple[float, float],
    target_range: tuple[float, float],
    output_path: Path,
) -> None:
    """Write the concise baseline evaluation section requested by the workflow."""
    output_path.write_text(
        f"""# Baseline NumPy Vanilla RNN Evaluation

## Baseline configuration

- Model: NumPy Vanilla RNN
- Input: `24 x 7`
- Hidden state: `h_t in R^32`
- Output: `y_hat in R`
- Optimizer: plain SGD
- Learning rate: `0.001`
- Epochs: `2`
- Random seed: `42`

The model was trained only on the chronological training set. The test set was
kept unseen: evaluation performs forward propagation only and does not update
any parameters. Predictions and targets were inverse-transformed with the
training-only `Global_active_power` min/max values from
`preprocessing_metadata.rds`. All reported metrics use the original kW unit.

## Test metrics

- MAE: `{metrics['MAE_kW']:.8f}` kW
- MSE: `{metrics['MSE_kW2']:.8f}` kW^2
- RMSE: `{metrics['RMSE_kW']:.8f}` kW
- R^2: `{metrics['R2']:.8f}`

Prediction range: `{prediction_range[0]:.8f}` to `{prediction_range[1]:.8f}` kW  
Actual target range: `{target_range[0]:.8f}` to `{target_range[1]:.8f}` kW

These results describe one specific baseline configuration and are not an
optimized-model claim. Training behavior, unseen-test metrics, the
actual-versus-predicted plot, and the signed-error plot should be considered
together.
""",
        encoding="utf-8",
    )


def evaluate(
    model_path: Path,
    test_path: Path,
    metadata_path: Path,
    results_dir: Path,
    rscript_path: str | None,
) -> dict[str, object]:
    """Evaluate the saved baseline and write all requested artifacts."""
    model = load_saved_model(model_path)
    X_test, y_test_scaled = load_rds_sequences(test_path, rscript_path)
    if X_test.shape != (6894, 24, 7) or y_test_scaled.shape != (6894,):
        raise ValueError(f"Unexpected test shapes: {X_test.shape}, {y_test_scaled.shape}")

    predictions_scaled = generate_predictions(model, X_test)
    if not np.isfinite(predictions_scaled).all() or not np.isfinite(y_test_scaled).all():
        raise FloatingPointError("NaN or Inf found in scaled predictions or targets")

    target_min, target_max = load_target_scaling(metadata_path, rscript_path)
    scale = target_max - target_min
    predictions_kw = predictions_scaled * scale + target_min
    actual_kw = y_test_scaled * scale + target_min
    metrics = calculate_metrics(actual_kw, predictions_kw)
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "rnn_test_predictions.npy"
    metrics_path = results_dir / "rnn_test_metrics.json"
    actual_plot_path = results_dir / "rnn_actual_vs_predicted.png"
    error_plot_path = results_dir / "rnn_prediction_error.png"
    csv_path = results_dir / "rnn_test_predictions.csv"
    report_path = results_dir / "rnn_baseline_evaluation.md"
    np.save(predictions_path, predictions_kw)
    save_evaluation_plot(actual_kw, predictions_kw, actual_plot_path)
    save_error_plot(actual_kw, predictions_kw, error_plot_path)
    save_predictions_csv(actual_kw, predictions_kw, csv_path)

    result_record = {
        "model": "NumPy Vanilla RNN",
        "hidden_units": 32,
        "timesteps": 24,
        "features": 7,
        "optimizer": "SGD",
        "learning_rate": 0.001,
        "epochs": 2,
        "random_seed": 42,
        "test_shape": {"X_test": list(X_test.shape), "y_test": list(y_test_scaled.shape)},
        "scaled_prediction_summary": {
            "minimum": float(predictions_scaled.min()),
            "maximum": float(predictions_scaled.max()),
            "mean": float(predictions_scaled.mean()),
            "target_minimum": float(y_test_scaled.min()),
            "target_maximum": float(y_test_scaled.max()),
            "target_mean": float(y_test_scaled.mean()),
            "finite": bool(np.isfinite(predictions_scaled).all()),
        },
        "target_scaling": {
            "source": "preprocessing/preprocessing_metadata.rds",
            "target": "Global_active_power",
            "minimum_kW": target_min,
            "maximum_kW": target_max,
            "inverse_transform_verified": True,
        },
        "original_scale_summary": {
            "prediction_minimum_kW": float(predictions_kw.min()),
            "prediction_maximum_kW": float(predictions_kw.max()),
            "prediction_mean_kW": float(predictions_kw.mean()),
            "actual_minimum_kW": float(actual_kw.min()),
            "actual_maximum_kW": float(actual_kw.max()),
            "actual_mean_kW": float(actual_kw.mean()),
        },
        "metrics": metrics,
        "artifacts": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "actual_vs_predicted_plot": str(actual_plot_path),
            "error_plot": str(error_plot_path),
            "predictions_csv": str(csv_path),
            "report": str(report_path),
        },
    }
    metrics_path.write_text(json.dumps(result_record, indent=2), encoding="utf-8")
    save_report(
        metrics,
        (float(predictions_kw.min()), float(predictions_kw.max())),
        (float(actual_kw.min()), float(actual_kw.max())),
        report_path,
    )
    print("Model parameters: W_xh, W_hh, b_h, W_hy, b_y")
    print("Test shape:", X_test.shape, y_test_scaled.shape)
    print("Scaled prediction range:", predictions_scaled.min(), predictions_scaled.max())
    print("Prediction range (kW):", predictions_kw.min(), predictions_kw.max())
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")
    print("Saved evaluation artifacts in:", results_dir)
    return result_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=Path, default=PROJECT_ROOT / "results" / "rnn_numpy_model.npz"
    )
    parser.add_argument(
        "--test-path", type=Path, default=PROJECT_ROOT / "preprocessing" / "test_sequences.rds"
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=PROJECT_ROOT / "preprocessing" / "preprocessing_metadata.rds",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--rscript", default=None)
    args = parser.parse_args()
    evaluate(args.model_path, args.test_path, args.metadata_path, args.results_dir, args.rscript)


if __name__ == "__main__":
    main()
