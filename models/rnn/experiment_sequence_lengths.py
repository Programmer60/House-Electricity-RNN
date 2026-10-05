"""Step 10: controlled sequence-length experiment for the PyTorch RNN."""

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
import torch
from torch import nn

try:
    from .evaluate_numpy import calculate_metrics, load_target_scaling
    from .rnn_pytorch import PyTorchVanillaRNN, load_rds_sequences, make_loader, run_epoch, set_reproducible_seed
except ImportError:
    from evaluate_numpy import calculate_metrics, load_target_scaling
    from rnn_pytorch import PyTorchVanillaRNN, load_rds_sequences, make_loader, run_epoch, set_reproducible_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTHS = (6, 12, 24, 48)


def predict(model: PyTorchVanillaRNN, X: np.ndarray) -> np.ndarray:
    """Generate predictions without parameter updates."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)


def parameter_count(input_size: int = 7, hidden_size: int = 32, output_size: int = 1) -> int:
    """Count parameters for this fixed 32-unit architecture."""
    return hidden_size * input_size + hidden_size * hidden_size + hidden_size + hidden_size * output_size + output_size


def train_one(
    sequence_length: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[PyTorchVanillaRNN, dict[str, list[float]], int, float, float]:
    """Train one sequence length using the locked PyTorch controls."""
    set_reproducible_seed(42)
    train_loader = make_loader(X_train, y_train, 64)
    validation_loader = make_loader(X_validation, y_validation, 64)
    model = PyTorchVanillaRNN(input_size=7, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = {"training_loss": [], "validation_loss": []}
    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    start = time.perf_counter()
    for epoch in range(1, 31):
        training_loss = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, None)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"T{sequence_length} epoch {epoch:02d}   "
            f"training={training_loss:.8f} validation={validation_loss:.8f}"
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
        if epochs_without_improvement >= 5:
            print(f"T{sequence_length} early stopping after epoch {epoch}")
            break
    runtime_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError(f"T{sequence_length} produced no validation checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_validation_loss, runtime_seconds


def load_sequences(sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load R-generated train, validation, and test RDS files."""
    prefix = PROJECT_ROOT / "preprocessing"
    X_train, y_train = load_rds_sequences(prefix / f"train_sequences_T{sequence_length}.rds")
    X_validation, y_validation = load_rds_sequences(prefix / f"validation_sequences_T{sequence_length}.rds")
    X_test, y_test = load_rds_sequences(prefix / f"test_sequences_T{sequence_length}.rds")
    if X_train.shape[1:] != (sequence_length, 7):
        raise ValueError(f"Unexpected T{sequence_length} train shape: {X_train.shape}")
    if X_validation.shape[1:] != (sequence_length, 7):
        raise ValueError(f"Unexpected T{sequence_length} validation shape: {X_validation.shape}")
    if X_test.shape[1:] != (sequence_length, 7):
        raise ValueError(f"Unexpected T{sequence_length} test shape: {X_test.shape}")
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def save_loss_plot(results: dict[str, dict[str, object]], output_path: Path) -> None:
    """Plot validation loss for all sequence lengths."""
    figure, axis = plt.subplots(figsize=(9, 5))
    for sequence_length in SEQUENCE_LENGTHS:
        history = results[str(sequence_length)]["history"]["validation_loss"]
        axis.plot(np.arange(1, len(history) + 1), history, marker="o", label=f"T={sequence_length}")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation MSE loss (scaled target)")
    axis.set_title("Validation loss by sequence length")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metric_plot(results: dict[str, dict[str, object]], metric: str, ylabel: str, output_path: Path) -> None:
    """Plot a test metric against sequence length."""
    values = [results[str(sequence_length)]["metrics"][metric] for sequence_length in SEQUENCE_LENGTHS]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(SEQUENCE_LENGTHS, values, marker="o")
    axis.set_xticks(SEQUENCE_LENGTHS)
    axis.set_xlabel("Sequence length (hours)")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} versus sequence length")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_prediction_plot(predictions: dict[str, np.ndarray], actual_by_length: dict[str, np.ndarray], output_path: Path) -> None:
    """Plot the first 500 predictions for all sequence lengths."""
    figure, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for axis, sequence_length in zip(axes, SEQUENCE_LENGTHS):
        actual = actual_by_length[str(sequence_length)]
        predicted = predictions[str(sequence_length)]
        count = min(500, actual.size)
        samples = np.arange(count)
        axis.plot(samples, actual[:count], label="Actual", linewidth=1.0)
        axis.plot(samples, predicted[:count], label=f"T={sequence_length}", linewidth=1.0)
        axis.set_ylabel("Power (kW)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Test sample")
    figure.suptitle("First 500 test predictions by sequence length")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_csv(results: dict[str, dict[str, object]], output_path: Path) -> None:
    """Save the requested sequence-length result table."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "sequence_length", "train_sequences", "validation_sequences", "test_sequences",
            "best_epoch", "best_validation_loss", "test_mae_kW", "test_rmse_kW",
            "test_r2", "runtime_seconds",
        ])
        for sequence_length in SEQUENCE_LENGTHS:
            result = results[str(sequence_length)]
            writer.writerow([
                sequence_length, result["train_sequences"], result["validation_sequences"],
                result["test_sequences"], result["best_epoch"], result["best_validation_loss"],
                result["metrics"]["MAE_kW"], result["metrics"]["RMSE_kW"],
                result["metrics"]["R2"], result["runtime_seconds"],
            ])


def save_report(results: dict[str, dict[str, object]], output_path: Path) -> None:
    """Write methodology, results, boundary handling, and interpretation."""
    rows = []
    for sequence_length in SEQUENCE_LENGTHS:
        result = results[str(sequence_length)]
        rows.append(
            f"| {sequence_length} hours | {result['train_sequences']} | {result['validation_sequences']} | "
            f"{result['test_sequences']} | {len(result['history']['training_loss'])} | {result['best_epoch']} | "
            f"{result['best_validation_loss']:.8f} | {result['metrics']['MAE_kW']:.8f} | "
            f"{result['metrics']['RMSE_kW']:.8f} | {result['metrics']['R2']:.8f} |"
        )
    best_rmse = min(SEQUENCE_LENGTHS, key=lambda length: results[str(length)]["metrics"]["RMSE_kW"])
    best_r2 = max(SEQUENCE_LENGTHS, key=lambda length: results[str(length)]["metrics"]["R2"])
    output_path.write_text(
        f"""# Step 10: Controlled Sequence-Length Experiment

## Methodology and controls

Alternative sequences were generated in R from the hourly preprocessing pipeline;
Python did not slice the existing 24-timestep arrays. The original
`train_sequences.rds` and `test_sequences.rds` remain unchanged.

The original chronological training period was scaled using its training-only
min/max values, then its sequence stream was split chronologically into a
training prefix and validation suffix. The original test period was generated
as a separate sequence stream. This preserves the locked 24-hour protocol and
prevents future observations from entering an input window. Validation windows
retain earlier training-period observations as historical input context. Test
windows intentionally restart at the test partition boundary, matching the
locked baseline convention; preceding-period values are not used as test input
context.

For every length, the sequence rule was `X[i] = rows i:(i+T-1)` and
`y[i] = Global_active_power at row i+T`. The generated counts are recorded
below. All PyTorch controls stayed fixed: 7 features, 32 hidden units, tanh
RNN, Adam, learning rate `0.001`, batch size `64`, seed `42`, no shuffling,
maximum 30 epochs, and patience 5. The test set was used only after selecting
the lowest-validation-loss checkpoint.

## Results

| Sequence Length | Train Sequences | Validation Sequences | Test Sequences | Actual Epochs | Best Epoch | Best Val Loss | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

The 24-hour result reproduced the locked RNN-32 baseline within the required
numerical tolerance. Its MAE, RMSE, and R^2 match the locked values.

## Mathematical interpretation

Increasing `T` requires the recurrence

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

to process more transitions: 6, 12, 24, or 48 recurrent steps. During BPTT,
this connects to repeated Jacobian products such as

```text
partial L / partial h_t is controlled by products of J_k^T
```

Longer sequences can expose gradients to more recurrent multiplications and
increase computation, but performance must be determined empirically.

## Interpretation

- Lowest test RMSE among the tested lengths: **T={best_rmse}**.
- Highest test R^2 among the tested lengths: **T={best_r2}**.
- The 24-hour history provides a useful measured reference point, but these
  four runs do not establish that it is universally optimal.
- Sequence length changes the number of input timesteps and therefore runtime,
  memory use, available historical context, and possible gradient-propagation
  distance. It does not automatically improve or degrade generalization.
- These conclusions apply only to T=6, 12, 24, and 48 under this fixed training
  configuration. No other experiments were run.
""",
        encoding="utf-8",
    )


def run_experiment(results_dir: Path, models_dir: Path) -> dict[str, object]:
    """Run the four R-generated sequence datasets with a T24 reproducibility gate."""
    target_min, target_max = load_target_scaling(PROJECT_ROOT / "preprocessing/preprocessing_metadata.rds")
    scale = target_max - target_min
    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    actual_by_length: dict[str, np.ndarray] = {}

    # Run the locked-length reproduction first. If it fails, do not accept any
    # other sequence-length result.
    run_order = (24, 6, 12, 48)
    for sequence_length in run_order:
        X_train, y_train, X_validation, y_validation, X_test, y_test = load_sequences(sequence_length)
        model, history, best_epoch, best_validation_loss, runtime_seconds = train_one(
            sequence_length, X_train, y_train, X_validation, y_validation
        )
        predictions_scaled = predict(model, X_test)
        predictions_kw = predictions_scaled * scale + target_min
        actual_kw = y_test * scale + target_min
        metrics = calculate_metrics(actual_kw, predictions_kw)
        if sequence_length == 24:
            locked = json.loads((results_dir / "rnn_pytorch_metrics.json").read_text(encoding="utf-8"))["test_metrics"]
            for metric in ("MAE_kW", "RMSE_kW", "R2"):
                if abs(metrics[metric] - locked[metric]) > 1e-5:
                    raise RuntimeError(
                        f"T24 failed locked-baseline reproduction for {metric}: "
                        f"{metrics[metric]} vs {locked[metric]}"
                    )
            print("T24 locked baseline reproduction: passed")
        predictions[str(sequence_length)] = predictions_kw
        actual_by_length[str(sequence_length)] = actual_kw
        results[str(sequence_length)] = {
            "sequence_length": sequence_length,
            "train_sequences": int(X_train.shape[0]),
            "validation_sequences": int(X_validation.shape[0]),
            "test_sequences": int(X_test.shape[0]),
            "history": history,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "metrics": metrics,
            "runtime_seconds": runtime_seconds,
            "prediction_distribution": {
                "minimum_kW": float(predictions_kw.min()),
                "maximum_kW": float(predictions_kw.max()),
                "mean_kW": float(predictions_kw.mean()),
            },
        }
        models_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "configuration": {
                    "input_size": 7,
                    "hidden_units": 32,
                    "timesteps": sequence_length,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "optimizer": "Adam",
                    "max_epochs": 30,
                    "patience": 5,
                    "seed": 42,
                    "shuffle": False,
                },
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
            },
            models_dir / f"rnn_T{sequence_length}_best.pt",
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    save_metric_plot(results, "RMSE_kW", "Test RMSE (kW)", results_dir / "rnn_sequence_length_rmse.png")
    save_metric_plot(results, "R2", "Test R^2", results_dir / "rnn_sequence_length_r2.png")
    save_loss_plot(results, results_dir / "rnn_sequence_length_validation_loss.png")
    save_prediction_plot(predictions, actual_by_length, results_dir / "rnn_sequence_length_predictions.png")
    save_csv(results, results_dir / "rnn_sequence_length_results.csv")
    save_report(results, results_dir / "rnn_sequence_length_experiment.md")
    aggregate = {
        "experiment": "controlled sequence-length experiment",
        "controls": {
            "input_size": 7,
            "hidden_units": 32,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "seed": 42,
            "shuffle": False,
            "maximum_epochs": 30,
            "patience": 5,
            "split": "chronological R-generated train/validation/test sequence streams",
            "scaling": "training-period min/max only",
        },
        "t24_locked_baseline_reproduced": True,
        "results": {str(length): results[str(length)] for length in SEQUENCE_LENGTHS},
    }
    (results_dir / "rnn_sequence_length_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    for sequence_length in SEQUENCE_LENGTHS:
        result = results[str(sequence_length)]
        print(
            f"T{sequence_length}: train={result['train_sequences']} "
            f"validation={result['validation_sequences']} test={result['test_sequences']} "
            f"best_epoch={result['best_epoch']} val_loss={result['best_validation_loss']:.8f} "
            f"MAE={result['metrics']['MAE_kW']:.8f} "
            f"RMSE={result['metrics']['RMSE_kW']:.8f} "
            f"R2={result['metrics']['R2']:.8f} runtime={result['runtime_seconds']:.2f}s"
        )
    print("Step 10 sequence-length experiment: complete")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    run_experiment(args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
