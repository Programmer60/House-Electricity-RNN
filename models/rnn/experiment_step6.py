"""Step 6: investigate longer SGD training for the NumPy vanilla RNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .evaluate_numpy import (
        calculate_metrics,
        load_saved_model,
        load_target_scaling,
        save_error_plot,
        save_evaluation_plot,
    )
    from .train_numpy import load_rds_sequences
    from .rnn_numpy import VanillaRNN
except ImportError:
    from evaluate_numpy import (
        calculate_metrics,
        load_saved_model,
        load_target_scaling,
        save_error_plot,
        save_evaluation_plot,
    )
    from train_numpy import load_rds_sequences
    from rnn_numpy import VanillaRNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_training_plot(history: list[float], output_path: Path) -> None:
    """Plot full-data training loss for the 20-epoch experiment."""
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(np.arange(1, len(history) + 1), history, marker="o")
    axis.set_title("Step 6: 20-epoch NumPy RNN training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss (1/2 squared error)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_comparison_table(
    baseline: dict[str, float],
    experiment: dict[str, float],
    output_path: Path,
) -> None:
    """Save the requested two-row baseline comparison."""
    output_path.write_text(
        """# Step 6 Training-Dynamics Comparison

| Configuration | Final Train Loss | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---|---:|---:|---:|---:|
| SGD, 2 epochs | {baseline_loss:.8f} | {baseline_mae:.8f} | {baseline_rmse:.8f} | {baseline_r2:.8f} |
| SGD, 20 epochs | {experiment_loss:.8f} | {experiment_mae:.8f} | {experiment_rmse:.8f} | {experiment_r2:.8f} |
""".format(
            baseline_loss=baseline["final_train_loss"],
            baseline_mae=baseline["MAE_kW"],
            baseline_rmse=baseline["RMSE_kW"],
            baseline_r2=baseline["R2"],
            experiment_loss=experiment["final_train_loss"],
            experiment_mae=experiment["MAE_kW"],
            experiment_rmse=experiment["RMSE_kW"],
            experiment_r2=experiment["R2"],
        ),
        encoding="utf-8",
    )


def save_report(
    results: dict[str, object],
    comparison: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    """Save the Step 6 interpretation and mathematical connection."""
    experiment = results["experiment_20_epochs"]
    distribution = experiment["distribution"]
    baseline = comparison["2_epochs"]
    report = f"""# Step 6: NumPy Vanilla RNN Training Dynamics

## Controlled experiment

The same validated vanilla RNN was retrained with only the epoch count changed:

- Input: `24 x 7`
- Hidden units: `32`
- Optimizer: plain SGD
- Learning rate: `0.001`
- Seed: `42`
- Sample order: chronological, no shuffling
- Epochs: `20`

The model parameters were initialized from the same seed for this experiment.
The unseen test set was used only after all 20 training epochs.

## Results

- Final training loss: `{experiment['final_train_loss']:.8f}`
- Test MAE: `{experiment['metrics']['MAE_kW']:.8f}` kW
- Test RMSE: `{experiment['metrics']['RMSE_kW']:.8f}` kW
- Test R^2: `{experiment['metrics']['R2']:.8f}`
- Runtime: `{experiment['runtime_seconds']:.2f}` seconds

Prediction distribution:

- Prediction range: `{distribution['prediction_minimum_kW']:.8f}` to `{distribution['prediction_maximum_kW']:.8f}` kW
- Prediction mean: `{distribution['prediction_mean_kW']:.8f}` kW
- Actual range: `{distribution['actual_minimum_kW']:.8f}` to `{distribution['actual_maximum_kW']:.8f}` kW
- Actual mean: `{distribution['actual_mean_kW']:.8f}` kW

The prediction band remains substantially narrower than the actual test range.
The comparison table should be read together with the plots: lower training
loss alone does not establish improved generalization.

## Mathematical connection

Each training sequence performs the validated forward pass, single-sample loss,
and BPTT gradient calculation. Plain SGD then applies:

```text
theta_new = theta_old - eta * grad_theta L
```

where:

```text
theta = {{W_xh, W_hh, b_h, W_hy, b_y}}
```

Twenty epochs provide more repeated parameter updates than the two-epoch
baseline. This experiment tests whether additional optimization reduces the
underfitting observed in the baseline without changing the model architecture.
"""
    output_path.write_text(report, encoding="utf-8")


def run_experiment(
    results_dir: Path,
    rscript_path: str | None = None,
) -> dict[str, object]:
    """Train and evaluate the exact 20-epoch controlled experiment."""
    train_path = PROJECT_ROOT / "preprocessing" / "train_sequences.rds"
    test_path = PROJECT_ROOT / "preprocessing" / "test_sequences.rds"
    metadata_path = PROJECT_ROOT / "preprocessing" / "preprocessing_metadata.rds"
    baseline_metrics_path = results_dir / "rnn_test_metrics.json"
    baseline_training_path = results_dir / "rnn_numpy_training_results.json"
    X_train, y_train = load_rds_sequences(train_path, rscript_path)
    X_test, y_test_scaled = load_rds_sequences(test_path, rscript_path)
    if X_train.shape != (27647, 24, 7) or y_train.shape != (27647,):
        raise ValueError(f"Unexpected train shapes: {X_train.shape}, {y_train.shape}")
    if X_test.shape != (6894, 24, 7) or y_test_scaled.shape != (6894,):
        raise ValueError(f"Unexpected test shapes: {X_test.shape}, {y_test_scaled.shape}")

    model = VanillaRNN(input_size=7, hidden_size=32, output_size=1, seed=42)
    start = perf_counter()
    history = model.train(
        X_train,
        y_train,
        epochs=20,
        learning_rate=0.001,
        print_progress=True,
    )
    runtime_seconds = perf_counter() - start

    predictions_scaled = np.array(
        [model.forward(sequence)[0][0] for sequence in X_test], dtype=float
    )
    target_min, target_max = load_target_scaling(metadata_path, rscript_path)
    scale = target_max - target_min
    predictions_kw = predictions_scaled * scale + target_min
    actual_kw = y_test_scaled * scale + target_min
    if not np.isfinite(predictions_scaled).all():
        raise FloatingPointError("NaN or Inf found in 20-epoch predictions")
    metrics = calculate_metrics(actual_kw, predictions_kw)

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "rnn_numpy_model_20_epochs.npz"
    np.savez(
        model_path,
        W_xh=model.W_xh,
        W_hh=model.W_hh,
        b_h=model.b_h,
        W_hy=model.W_hy,
        b_y=model.b_y,
    )
    predictions_path = results_dir / "rnn_test_predictions_20_epochs.npy"
    training_plot_path = results_dir / "rnn_20_epoch_training_loss.png"
    actual_plot_path = results_dir / "rnn_20_epoch_actual_vs_predicted.png"
    error_plot_path = results_dir / "rnn_20_epoch_prediction_error.png"
    np.save(predictions_path, predictions_kw)
    save_training_plot(history, training_plot_path)
    save_evaluation_plot(actual_kw, predictions_kw, actual_plot_path)
    save_error_plot(actual_kw, predictions_kw, error_plot_path)

    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_training = json.loads(baseline_training_path.read_text(encoding="utf-8"))
    comparison = {
        "2_epochs": {
            "final_train_loss": baseline_training["full_training"]["final_loss"],
            **baseline_metrics["metrics"],
        },
        "20_epochs": {
            "final_train_loss": history[-1],
            **metrics,
        },
    }
    comparison_path = results_dir / "rnn_step6_comparison.md"
    save_comparison_table(comparison["2_epochs"], comparison["20_epochs"], comparison_path)

    result_record = {
        "experiment": "Step 6 longer SGD training",
        "configuration": {
            "hidden_units": 32,
            "timesteps": 24,
            "features": 7,
            "optimizer": "SGD",
            "learning_rate": 0.001,
            "epochs": 20,
            "random_seed": 42,
            "shuffle": False,
        },
        "experiment_20_epochs": {
            "history": history,
            "final_train_loss": history[-1],
            "runtime_seconds": runtime_seconds,
            "metrics": metrics,
            "distribution": {
                "prediction_minimum_kW": float(predictions_kw.min()),
                "prediction_maximum_kW": float(predictions_kw.max()),
                "prediction_mean_kW": float(predictions_kw.mean()),
                "actual_minimum_kW": float(actual_kw.min()),
                "actual_maximum_kW": float(actual_kw.max()),
                "actual_mean_kW": float(actual_kw.mean()),
            },
            "scaled_prediction_range": [
                float(predictions_scaled.min()),
                float(predictions_scaled.max()),
            ],
        },
        "comparison": comparison,
        "artifacts": {
            "model": str(model_path),
            "predictions": str(predictions_path),
            "training_plot": str(training_plot_path),
            "actual_vs_predicted_plot": str(actual_plot_path),
            "error_plot": str(error_plot_path),
            "comparison_table": str(comparison_path),
        },
    }
    results_path = results_dir / "rnn_step6_results.json"
    report_path = results_dir / "rnn_step6_report.md"
    results_path.write_text(json.dumps(result_record, indent=2), encoding="utf-8")
    save_report(result_record, comparison, report_path)
    print("Saved Step 6 results:", results_path)
    print("Saved Step 6 report:", report_path)
    print("20-epoch runtime seconds:", runtime_seconds)
    print("20-epoch final training loss:", history[-1])
    for name, value in metrics.items():
        print(f"20-epoch {name}: {value:.8f}")
    print("20-epoch prediction range (kW):", predictions_kw.min(), predictions_kw.max())
    return result_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--rscript", default=None)
    args = parser.parse_args()
    run_experiment(args.results_dir, args.rscript)


if __name__ == "__main__":
    main()
