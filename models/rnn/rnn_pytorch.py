"""Practical PyTorch vanilla RNN baseline for household power forecasting."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .evaluate_numpy import calculate_metrics, load_target_scaling
    from .train_numpy import load_rds_sequences
except ImportError:
    from evaluate_numpy import calculate_metrics, load_target_scaling
    from train_numpy import load_rds_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PyTorchVanillaRNN(nn.Module):
    """A many-to-one vanilla RNN matching the NumPy reference architecture."""

    def __init__(self, input_size: int = 7, hidden_size: int = 32) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            nonlinearity="tanh",
        )
        self.linear = nn.Linear(hidden_size, 1)     # Many-to-one output for next-hour prediction

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:   # Forward pass through the RNN and linear layer
        hidden_sequence, _ = self.rnn(sequence)
        final_hidden = hidden_sequence[:, -1, :]
        return self.linear(final_hidden)


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     # Set the seed for CPU and GPU (if available) operations


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader:
    """Create tensors with X=(N,24,7) and y=(N,1), preserving order."""
    features = torch.as_tensor(X, dtype=torch.float32)
    targets = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    return DataLoader(
        TensorDataset(features, targets),
        batch_size=batch_size,
        shuffle=False,
    )


def run_epoch(
    model: PyTorchVanillaRNN,
    loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one training or evaluation epoch and return mean MSE."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    sample_count = 0
    for features, targets in loader:
        if training:
            optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = features.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size
    return total_loss / sample_count


def predict(model: PyTorchVanillaRNN, X: np.ndarray) -> np.ndarray:
    """Generate predictions without gradients or parameter updates."""
    model.eval()
    features = torch.as_tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(features).cpu().numpy().reshape(-1)


def save_loss_plot(history: dict[str, list[float]], output_path: Path) -> None:
    """Save training and validation normalized MSE histories together."""
    epochs = np.arange(1, len(history["training_loss"]) + 1)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, history["training_loss"], marker="o", label="Training loss")
    axis.plot(epochs, history["validation_loss"], marker="o", label="Validation loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss (scaled target)")
    axis.set_title("PyTorch vanilla RNN training and validation loss")
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
    """Save first-500 time-series predictions and the actual/predicted scatter."""
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
    axis.set_title("PyTorch vanilla RNN prediction error")
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
    """Save actual, predicted, and signed error values in kW."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_kW", "predicted_kW", "error_kW"])
        writer.writerows(
            (actual, predicted, actual - predicted)
            for actual, predicted in zip(actual_kw, predicted_kw)
        )


def load_numpy_comparison(results_dir: Path) -> dict[str, dict[str, float]]:
    """Read the already measured NumPy baselines for the comparison table."""
    numpy_training = json.loads(
        (results_dir / "rnn_numpy_training_results.json").read_text(encoding="utf-8")
    )
    numpy_metrics = json.loads(
        (results_dir / "rnn_test_metrics.json").read_text(encoding="utf-8")
    )
    step6 = json.loads(
        (results_dir / "rnn_step6_results.json").read_text(encoding="utf-8")
    )
    return {
        "NumPy RNN, 2 epochs": {
            "epochs": 2,
            "optimizer": "SGD",
            "MAE_kW": numpy_metrics["metrics"]["MAE_kW"],
            "RMSE_kW": numpy_metrics["metrics"]["RMSE_kW"],
            "R2": numpy_metrics["metrics"]["R2"],
        },
        "NumPy RNN, 20 epochs": {
            "epochs": 20,
            "optimizer": "SGD",
            "MAE_kW": step6["experiment_20_epochs"]["metrics"]["MAE_kW"],
            "RMSE_kW": step6["experiment_20_epochs"]["metrics"]["RMSE_kW"],
            "R2": step6["experiment_20_epochs"]["metrics"]["R2"],
        },
    }


def save_comparison(
    comparison: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save measured NumPy and PyTorch baseline metrics."""
    lines = [
        "# NumPy and PyTorch Vanilla RNN Comparison",
        "",
        "| Model | Epochs | Optimizer | Test MAE (kW) | Test RMSE (kW) | Test R^2 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name, values in comparison.items():
        lines.append(
            f"| {name} | {values['epochs']} | {values['optimizer']} | "
            f"{values['MAE_kW']:.8f} | {values['RMSE_kW']:.8f} | {values['R2']:.8f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_report(
    result: dict[str, object],
    output_path: Path,
) -> None:
    """Save the practical-vs-reference mathematical correspondence."""
    metrics = result["test_metrics"]
    distribution = result["prediction_distribution"]
    output_path.write_text(
        f"""# Practical PyTorch Vanilla RNN Baseline

## Architecture and data split

The PyTorch model uses `nn.RNN(input_size=7, hidden_size=32,
batch_first=True, nonlinearity=\"tanh\")`, followed by `nn.Linear(32, 1)`. Only
the final output timestep is used for next-hour prediction.

The original chronological training set was split into:

- Training: first `{result['split']['train_samples']}` sequences
- Validation: final `{result['split']['validation_samples']}` sequences from the training period
- Test: untouched `{result['split']['test_samples']}` sequences

No preprocessing or normalization was redone. The final test evaluation was
performed after selecting the lowest-validation-loss checkpoint.

## Configuration

- Batch size: `{result['configuration']['batch_size']}`
- Learning rate: `{result['configuration']['learning_rate']}`
- Optimizer: Adam
- Maximum epochs: `{result['configuration']['max_epochs']}`
- Epochs actually trained: `{result['epochs_trained']}`
- Patience: `{result['configuration']['patience']}`
- Seed: `{result['configuration']['seed']}`
- Best epoch: `{result['best_epoch']}`
- Best validation loss: `{result['best_validation_loss']:.8f}`
- Runtime: `{result['runtime_seconds']:.2f}` seconds

## Test results

- MAE: `{metrics['MAE_kW']:.8f}` kW
- MSE: `{metrics['MSE_kW2']:.8f}` kW^2
- RMSE: `{metrics['RMSE_kW']:.8f}` kW
- R^2: `{metrics['R2']:.8f}`
- Prediction range: `{distribution['minimum_kW']:.8f}` to `{distribution['maximum_kW']:.8f}` kW

The predictions were inverse-transformed using the training-only target scaling
parameters. No test sequence was used for optimization or checkpoint selection.

## Mathematical correspondence

The PyTorch recurrent layer automates the same computation represented manually
in NumPy:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
y_hat = W_hy h_T + b_y
L = mean((y - y_hat)^2)
```

The NumPy implementation explicitly derives BPTT and performs plain SGD:

```text
theta_new = theta_old - eta * grad_theta L
```

The PyTorch implementation uses autograd for the gradients and Adam for the
parameter updates. Adam is a practical optimization choice, not a mathematical
requirement of the RNN.
""",
        encoding="utf-8",
    )


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
    """Train with chronological validation and evaluate the untouched test set."""
    set_reproducible_seed(seed)
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
    X_train = X_train_all[:split_index]
    y_train = y_train_all[:split_index]
    X_validation = X_train_all[split_index:]
    y_validation = y_train_all[split_index:]
    train_loader = make_loader(X_train, y_train, batch_size)
    validation_loader = make_loader(X_validation, y_validation, batch_size)

    model = PyTorchVanillaRNN(input_size=7, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam is a practical choice for optimization, not a mathematical requirement of the RNN
    history = {"training_loss": [], "validation_loss": []}
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    start_time = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        training_loss = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, None)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"Epoch {epoch:02d}   Training loss: {training_loss:.8f}   "
            f"Validation loss: {validation_loss:.8f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping after epoch {epoch}")
            break
    runtime_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("No validation checkpoint was recorded")
    model.load_state_dict(best_state)

    predictions_scaled = predict(model, X_test)
    target_min, target_max = load_target_scaling(metadata_path)
    scale = target_max - target_min
    predictions_kw = predictions_scaled * scale + target_min
    actual_kw = y_test_scaled * scale + target_min
    if not np.isfinite(predictions_kw).all():
        raise FloatingPointError("NaN or Inf found in PyTorch predictions")
    metrics = calculate_metrics(actual_kw, predictions_kw)

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "rnn_pytorch_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "configuration": {
                "input_size": 7,
                "hidden_units": 32,
                "timesteps": 24,
            },
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
        },
        model_path,
    )
    predictions_path = results_dir / "rnn_pytorch_predictions.csv"
    loss_plot_path = results_dir / "rnn_pytorch_loss.png"
    actual_plot_path = results_dir / "rnn_pytorch_actual_vs_predicted.png"
    error_plot_path = results_dir / "rnn_pytorch_error.png"
    save_predictions_csv(actual_kw, predictions_kw, predictions_path)
    save_loss_plot(history, loss_plot_path)
    save_prediction_plot(actual_kw, predictions_kw, actual_plot_path)
    save_error_plot(actual_kw, predictions_kw, error_plot_path)

    comparison = load_numpy_comparison(results_dir)
    comparison["PyTorch RNN"] = {
        "epochs": best_epoch,
        "optimizer": "Adam",
        **metrics,
    }
    comparison_path = results_dir / "rnn_pytorch_comparison.md"
    save_comparison(comparison, comparison_path)

    result = {
        "model": "PyTorch Vanilla RNN",
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
            "description": "first 90% of training period for training; final 10% for validation",
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
            "actual_minimum_kW": float(actual_kw.min()),
            "actual_maximum_kW": float(actual_kw.max()),
            "actual_mean_kW": float(actual_kw.mean()),
        },
        "runtime_seconds": runtime_seconds,
        "comparison": comparison,
        "numerical_issues": {
            "predictions_finite": bool(np.isfinite(predictions_kw).all()),
            "training_history_finite": bool(
                np.isfinite(history["training_loss"]).all()
                and np.isfinite(history["validation_loss"]).all()
            ),
        },
        "artifacts": {
            "model": str(model_path),
            "metrics": str(results_dir / "rnn_pytorch_metrics.json"),
            "config": str(results_dir / "rnn_pytorch_config.json"),
            "predictions": str(predictions_path),
            "loss_plot": str(loss_plot_path),
            "actual_vs_predicted_plot": str(actual_plot_path),
            "error_plot": str(error_plot_path),
            "comparison": str(comparison_path),
            "report": str(results_dir / "rnn_pytorch_report.md"),
        },
    }
    (results_dir / "rnn_pytorch_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (results_dir / "rnn_pytorch_config.json").write_text(
        json.dumps(result["configuration"], indent=2), encoding="utf-8"
    )
    save_report(result, results_dir / "rnn_pytorch_report.md")
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_validation_loss)
    print("Runtime seconds:", runtime_seconds)
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")
    print("Prediction range (kW):", predictions_kw.min(), predictions_kw.max())
    print("Saved PyTorch artifacts in:", results_dir)
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
