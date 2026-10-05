"""Step 9: controlled hidden-unit experiment for the locked PyTorch RNN."""

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
    from .rnn_pytorch import (
        PyTorchVanillaRNN,
        load_rds_sequences,
        make_loader,
        run_epoch,
        set_reproducible_seed,
    )
except ImportError:
    from evaluate_numpy import calculate_metrics, load_target_scaling
    from rnn_pytorch import (
        PyTorchVanillaRNN,
        load_rds_sequences,
        make_loader,
        run_epoch,
        set_reproducible_seed,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HIDDEN_UNITS = (16, 32, 64)


def predict(model: PyTorchVanillaRNN, X: np.ndarray) -> np.ndarray:
    """Generate predictions without updating model parameters."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy().reshape(-1)


def parameter_count(input_size: int, hidden_size: int, output_size: int = 1) -> int:
    """Count RNN and linear-layer scalar parameters."""
    return hidden_size * input_size + hidden_size * hidden_size + hidden_size + hidden_size * output_size + output_size


def train_one(
    hidden_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[PyTorchVanillaRNN, dict[str, list[float]], int, float, float]:
    """Train one hidden size using the locked baseline controls."""
    set_reproducible_seed(seed)
    train_loader = make_loader(X_train, y_train, batch_size)
    validation_loader = make_loader(X_validation, y_validation, batch_size)
    model = PyTorchVanillaRNN(input_size=7, hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"training_loss": [], "validation_loss": []}
    best_state = None
    best_epoch = 0
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        training_loss = run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, None)
        history["training_loss"].append(training_loss)
        history["validation_loss"].append(validation_loss)
        print(
            f"RNN-{hidden_size} epoch {epoch:02d}   "
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
        if epochs_without_improvement >= patience:
            print(f"RNN-{hidden_size} early stopping after epoch {epoch}")
            break
    runtime_seconds = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError(f"RNN-{hidden_size} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_validation_loss, runtime_seconds


def save_loss_plot(results: dict[str, dict[str, object]], output_path: Path) -> None:
    """Plot training and validation curves for all hidden sizes."""
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for hidden_size in HIDDEN_UNITS:
        history = results[str(hidden_size)]["history"]
        epochs = np.arange(1, len(history["training_loss"]) + 1)
        axes[0].plot(epochs, history["training_loss"], marker="o", label=f"RNN-{hidden_size}")
        axes[1].plot(epochs, history["validation_loss"], marker="o", label=f"RNN-{hidden_size}")
    axes[0].set_title("Training loss by hidden size")
    axes[1].set_title("Validation loss by hidden size")
    axes[1].set_xlabel("Epoch")
    for axis in axes:
        axis.set_ylabel("MSE loss (scaled target)")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_prediction_plot(
    predictions: dict[str, np.ndarray],
    actual_kw: np.ndarray,
    output_path: Path,
) -> None:
    """Plot first-500 actual and predictions for each hidden size."""
    count = min(500, actual_kw.size)
    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    samples = np.arange(count)
    for axis, hidden_size in zip(axes, HIDDEN_UNITS):
        axis.plot(samples, actual_kw[:count], label="Actual", linewidth=1.0)
        axis.plot(samples, predictions[str(hidden_size)][:count], label=f"RNN-{hidden_size}", linewidth=1.0)
        axis.set_ylabel("Power (kW)")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Test sample")
    figure.suptitle("First 500 unseen test samples by hidden size")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_metric_plot(
    results: dict[str, dict[str, object]],
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot a test metric against hidden units."""
    values = [results[str(hidden_size)]["metrics"][metric] for hidden_size in HIDDEN_UNITS]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(HIDDEN_UNITS, values, marker="o")
    axis.set_xlabel("Hidden units")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} versus hidden units")
    axis.set_xticks(HIDDEN_UNITS)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_csv(results: dict[str, dict[str, object]], output_path: Path) -> None:
    """Save the requested aggregate result table."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model",
            "hidden_units",
            "parameters",
            "best_epoch",
            "best_validation_loss",
            "test_mae_kW",
            "test_rmse_kW",
            "test_r2",
            "runtime_seconds",
        ])
        for hidden_size in HIDDEN_UNITS:
            result = results[str(hidden_size)]
            writer.writerow([
                f"RNN-{hidden_size}",
                hidden_size,
                result["parameters"],
                result["best_epoch"],
                result["best_validation_loss"],
                result["metrics"]["MAE_kW"],
                result["metrics"]["RMSE_kW"],
                result["metrics"]["R2"],
                result["runtime_seconds"],
            ])


def save_report(
    results: dict[str, dict[str, object]],
    output_path: Path,
    baseline: dict[str, float],
) -> None:
    """Write the controlled-experiment interpretation."""
    rows = []
    parameter_rows = []
    for hidden_size in HIDDEN_UNITS:
        result = results[str(hidden_size)]
        rows.append(
            f"| RNN-{hidden_size} | {hidden_size} | {result['best_epoch']} | "
            f"{result['best_validation_loss']:.8f} | {result['metrics']['MAE_kW']:.8f} | "
            f"{result['metrics']['RMSE_kW']:.8f} | {result['metrics']['R2']:.8f} |"
        )
        parameter_rows.append(f"| {hidden_size} | {result['parameters']} |")
    rnn32 = results["32"]
    output_path.write_text(
        f"""# Step 9: Controlled Hidden-Unit Experiment

## Experimental control

Only the hidden dimension changed across RNN-16, RNN-32, and RNN-64. All runs
used the same input features, 24 timesteps, chronological data split, Adam
optimizer, learning rate `0.001`, batch size `64`, seed `42`, no shuffling,
maximum 30 epochs, and patience 5. The test set was not used for checkpoint
selection.

The split was identical for every run: `24,882` training sequences, `2,765`
validation sequences, and `6,894` untouched test sequences. Reusing seed `42`
controls randomization, but different hidden sizes necessarily create different
parameter tensor dimensions, so initial parameter arrays are not identical
element-for-element.

## Results

| Model | Hidden Units | Best Epoch | Best Val Loss | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Parameter counts

For input dimension `d=7`, hidden dimension `H`, and output dimension `1`:

```text
H*d + H^2 + H + H*1 + 1 = H(7) + H^2 + 2H + 1
```

| Hidden Units | Parameters |
|---:|---:|
{chr(10).join(parameter_rows)}

The recurrent matrix `W_hh` alone grows from `H x H`, so increasing hidden
units raises both representational capacity and computational/storage cost.

## Interpretation

- Validation loss was lowest for RNN-{min(HIDDEN_UNITS, key=lambda h: results[str(h)]['best_validation_loss'])} among the tested configurations.
- Test RMSE was lowest for RNN-{min(HIDDEN_UNITS, key=lambda h: results[str(h)]['metrics']['RMSE_kW'])} among the tested configurations.
- Test R^2 was highest for RNN-{max(HIDDEN_UNITS, key=lambda h: results[str(h)]['metrics']['R2'])} among the tested configurations.
- RNN-32 reproduced the locked baseline within the audit tolerance: recorded MAE `{baseline['MAE_kW']:.8f}` versus experiment `{rnn32['metrics']['MAE_kW']:.8f}`, recorded RMSE `{baseline['RMSE_kW']:.8f}` versus experiment `{rnn32['metrics']['RMSE_kW']:.8f}`, and recorded R^2 `{baseline['R2']:.8f}` versus experiment `{rnn32['metrics']['R2']:.8f}`.
- Training and validation curves should be considered together when assessing overfitting; lower training loss alone does not establish better generalization.

These conclusions apply only to the three tested hidden dimensions and this
fixed training configuration. They do not establish a universal optimum.
""",
        encoding="utf-8",
    )


def run_experiment(results_dir: Path, models_dir: Path) -> dict[str, object]:
    """Run exactly RNN-16, RNN-32, and RNN-64 under locked controls."""
    X_train_all, y_train_all = load_rds_sequences(PROJECT_ROOT / "preprocessing" / "train_sequences.rds")
    X_test, y_test_scaled = load_rds_sequences(PROJECT_ROOT / "preprocessing" / "test_sequences.rds")
    assert X_train_all.shape == (27647, 24, 7)
    assert y_train_all.shape == (27647,)
    assert X_test.shape == (6894, 24, 7)
    assert y_test_scaled.shape == (6894,)
    validation_samples = 2765
    split_index = 24882
    X_train, y_train = X_train_all[:split_index], y_train_all[:split_index]
    X_validation, y_validation = X_train_all[split_index:], y_train_all[split_index:]
    target_min, target_max = load_target_scaling(
        PROJECT_ROOT / "preprocessing" / "preprocessing_metadata.rds"
    )
    scale = target_max - target_min
    actual_kw = y_test_scaled * scale + target_min
    results: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    for hidden_size in HIDDEN_UNITS:
        model, history, best_epoch, best_validation_loss, runtime_seconds = train_one(
            hidden_size,
            X_train,
            y_train,
            X_validation,
            y_validation,
            batch_size=64,
            learning_rate=0.001,
            max_epochs=30,
            patience=5,
            seed=42,
        )
        predictions_scaled = predict(model, X_test)
        predictions_kw = predictions_scaled * scale + target_min
        metrics = calculate_metrics(actual_kw, predictions_kw)
        predictions[str(hidden_size)] = predictions_kw
        results[str(hidden_size)] = {
            "model": f"RNN-{hidden_size}",
            "hidden_units": hidden_size,
            "parameters": parameter_count(7, hidden_size, 1),
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "history": history,
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
                    "hidden_units": hidden_size,
                    "timesteps": 24,
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
            models_dir / f"rnn_{hidden_size}_best.pt",
        )

    locked_baseline = json.loads(
        (results_dir / "rnn_pytorch_metrics.json").read_text(encoding="utf-8")
    )["test_metrics"]
    rnn32 = results["32"]["metrics"]
    for metric in ("MAE_kW", "RMSE_kW", "R2"):
        if abs(rnn32[metric] - locked_baseline[metric]) > 1e-5:
            raise RuntimeError(
                f"RNN-32 failed locked-baseline reproducibility for {metric}: "
                f"{rnn32[metric]} vs {locked_baseline[metric]}"
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    save_loss_plot(results, results_dir / "rnn_hidden_units_loss.png")
    save_prediction_plot(predictions, actual_kw, results_dir / "rnn_hidden_units_actual_vs_predicted.png")
    save_metric_plot(results, "RMSE_kW", "Test RMSE (kW)", results_dir / "rnn_hidden_units_rmse.png")
    save_metric_plot(results, "R2", "Test R^2", results_dir / "rnn_hidden_units_r2.png")
    save_csv(results, results_dir / "rnn_hidden_units_results.csv")
    save_report(results, results_dir / "rnn_hidden_units_experiment.md", locked_baseline)
    aggregate = {
        "experiment": "controlled hidden-unit experiment",
        "controls": {
            "input_size": 7,
            "timesteps": 24,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 64,
            "seed": 42,
            "shuffle": False,
            "validation_split": "chronological 24882/2765 from training set",
            "test_samples": 6894,
            "max_epochs": 30,
            "patience": 5,
        },
        "locked_baseline_reproduced": True,
        "results": results,
    }
    (results_dir / "rnn_hidden_units_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    for hidden_size in HIDDEN_UNITS:
        result = results[str(hidden_size)]
        print(
            f"RNN-{hidden_size}: parameters={result['parameters']} "
            f"best_epoch={result['best_epoch']} "
            f"val_loss={result['best_validation_loss']:.8f} "
            f"MAE={result['metrics']['MAE_kW']:.8f} "
            f"RMSE={result['metrics']['RMSE_kW']:.8f} "
            f"R2={result['metrics']['R2']:.8f} "
            f"runtime={result['runtime_seconds']:.2f}s"
        )
    print("RNN-32 locked baseline reproduction: passed")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    run_experiment(args.results_dir, args.models_dir)


if __name__ == "__main__":
    main()
