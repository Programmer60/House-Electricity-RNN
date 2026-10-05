"""Training runner for the mathematically transparent NumPy vanilla RNN."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .rnn_numpy import VanillaRNN
except ImportError:
    from rnn_numpy import VanillaRNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_rscript(explicit_path: str | None = None) -> str:
    """Find Rscript without changing the R preprocessing pipeline."""
    candidates = [explicit_path, os.environ.get("RSCRIPT"), shutil.which("Rscript")]
    candidates.extend(str(path) for path in Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "Rscript was not found. Pass --rscript with the R installation path."
    )


def load_rds_sequences(path: Path, rscript_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load an R list(X, y) as NumPy arrays without redoing preprocessing.

    R arrays are column-major, so X is reshaped with order='F' to preserve
    the existing (samples, timesteps, features) contract.
    """
    rscript = find_rscript(rscript_path)
    output_path = Path(tempfile.mktemp(prefix="rnn_sequences_", suffix=".bin"))
    expression = (
        "args <- commandArgs(trailingOnly=TRUE); "
        "obj <- readRDS(args[1]); "
        "dims <- dim(obj$X); "
        "con <- file(args[2], open='wb'); "
        "writeBin(as.integer(dims), con, size=4, endian='little'); "
        "writeBin(as.double(obj$X), con, size=8, endian='little'); "
        "writeBin(as.double(obj$y), con, size=8, endian='little'); "
        "close(con)"
    )
    try:
        subprocess.run(
            [rscript, "-e", expression, str(path), str(output_path)],
            check=True,
            capture_output=True,
        )
        raw = output_path.read_bytes()
    finally:
        output_path.unlink(missing_ok=True)

    dimensions = tuple(np.frombuffer(raw[:12], dtype="<i4").astype(int))
    sample_count, timesteps, feature_count = dimensions
    value_count = sample_count * timesteps * feature_count
    values = np.frombuffer(raw[12:], dtype="<f8")
    X_train = values[:value_count].reshape(dimensions, order="F").copy()
    y_train = values[value_count:].copy()
    if y_train.shape != (sample_count,):
        raise ValueError(f"Unexpected y shape: {y_train.shape}")
    return X_train, y_train


def run_small_training_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    seed: int,
) -> tuple[list[float], float]:
    """Train on 200 chronological samples for a quick learning check."""
    sample_count = min(200, X_train.shape[0])
    model = VanillaRNN(X_train.shape[2], hidden_size=32, seed=seed)
    history = model.train(
        X_train[:sample_count],
        y_train[:sample_count],
        epochs=5,
        learning_rate=learning_rate,
    )
    if history[-1] >= history[0]:
        raise RuntimeError("Tiny-dataset loss did not decrease")
    return history, history[-1]


def run_overfit_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float,
    seed: int,
) -> tuple[list[float], float]:
    """Train repeatedly on 12 samples to verify the implementation can overfit."""
    sample_count = min(12, X_train.shape[0])
    model = VanillaRNN(X_train.shape[2], hidden_size=32, seed=seed)
    # Use a higher rate only for this debugging probe; the baseline stays at 0.001.
    overfit_learning_rate = learning_rate * 10.0
    history = model.train(
        X_train[:sample_count],
        y_train[:sample_count],
        epochs=500,
        learning_rate=overfit_learning_rate,
        print_progress=False,
    )
    if history[-1] >= history[0] * 0.25:
        raise RuntimeError("Overfit sanity check did not substantially reduce loss")
    return history, history[-1]


def save_loss_plot(histories: dict[str, list[float]], output_path: Path) -> None:
    """Save the three training histories as one clearly labeled plot."""
    plt.figure(figsize=(8, 5))
    for label, history in histories.items():
        plt.plot(np.arange(1, len(history) + 1), history, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (1/2 squared error)")
    plt.title("NumPy vanilla RNN training loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_training_suite(
    train_path: Path,
    results_dir: Path,
    full_epochs: int,
    learning_rate: float,
    seed: int,
    rscript_path: str | None,
) -> dict[str, object]:
    """Run sanity checks and the initial full-data baseline."""
    X_train, y_train = load_rds_sequences(train_path, rscript_path)
    if X_train.shape != (27647, 24, 7) or y_train.shape != (27647,):
        raise ValueError(f"Unexpected training shapes: {X_train.shape}, {y_train.shape}")

    tiny_history, tiny_final_loss = run_small_training_test(
        X_train, y_train, learning_rate, seed
    )
    overfit_history, overfit_final_loss = run_overfit_test(
        X_train, y_train, learning_rate, seed
    )

    full_model = VanillaRNN(input_size=7, hidden_size=32, seed=seed)
    start_time = time.perf_counter()
    full_history = full_model.train(
        X_train,
        y_train,
        epochs=full_epochs,
        learning_rate=learning_rate,
    )
    full_runtime_seconds = time.perf_counter() - start_time

    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "rnn_numpy_model.npz"
    np.savez(
        model_path,
        W_xh=full_model.W_xh,
        W_hh=full_model.W_hh,
        b_h=full_model.b_h,
        W_hy=full_model.W_hy,
        b_y=full_model.b_y,
    )
    plot_path = results_dir / "rnn_numpy_loss.png"
    histories = {
        "tiny (200 samples)": tiny_history,
        "overfit (12 samples)": overfit_history,
        "full training": full_history,
    }
    save_loss_plot(histories, plot_path)

    results = {
        "configuration": {
            "hidden_units": 32,
            "timesteps": 24,
            "features": 7,
            "learning_rate": learning_rate,
            "epochs": full_epochs,
            "optimizer": "plain stochastic gradient descent",
            "loss_function": "0.5 * (y - y_hat)^2 per sample",
            "random_seed": seed,
            "shuffle": False,
        },
        "data_shape": {"X_train": list(X_train.shape), "y_train": list(y_train.shape)},
        "tiny_test": {
            "sample_count": 200,
            "epochs": len(tiny_history),
            "initial_loss": tiny_history[0],
            "final_loss": tiny_final_loss,
            "history": tiny_history,
        },
        "overfit_test": {
            "sample_count": 12,
            "epochs": len(overfit_history),
            "learning_rate": learning_rate * 10.0,
            "initial_loss": overfit_history[0],
            "final_loss": overfit_final_loss,
            "history": overfit_history,
        },
        "full_training": {
            "runtime_seconds": full_runtime_seconds,
            "initial_loss": full_history[0],
            "final_loss": full_history[-1],
            "history": full_history,
            "finite_history": bool(np.isfinite(full_history).all()),
        },
        "artifacts": {"model": str(model_path), "loss_plot": str(plot_path)},
    }
    results_path = results_dir / "rnn_numpy_training_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results: {results_path}")
    print(f"Saved loss plot: {plot_path}")
    print(f"Saved model: {model_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rscript", default=None)
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "preprocessing" / "train_sequences.rds",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    args = parser.parse_args()
    run_training_suite(
        args.train_path,
        args.results_dir,
        args.full_epochs,
        args.learning_rate,
        args.seed,
        args.rscript,
    )


if __name__ == "__main__":
    main()
