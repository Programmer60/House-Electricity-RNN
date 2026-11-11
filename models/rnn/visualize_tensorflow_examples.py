"""Visualize the trained Keras RNN architecture and real test-set predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

try:
    from .evaluate_numpy import load_target_scaling
    from .train_numpy import load_rds_sequences
except ImportError:
    from evaluate_numpy import load_target_scaling
    from train_numpy import load_rds_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_NAMES = (
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
)


def add_box(axis: plt.Axes, x: float, y: float, width: float, height: float, title: str, detail: str) -> None:
    """Draw a labeled architecture component."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor="#e8f1fb",
        edgecolor="#28527a",
        linewidth=1.5,
    )
    axis.add_patch(box)
    axis.text(x + width / 2, y + height * 0.65, title, ha="center", va="center", weight="bold")
    axis.text(x + width / 2, y + height * 0.3, detail, ha="center", va="center", fontsize=9)


def add_arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str | None = None) -> None:
    """Draw a directional arrow between architecture components."""
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#333333")
    )
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.08, label, ha="center", fontsize=9)


def save_architecture_plot(output_path: Path) -> None:
    """Draw the actual 24x7 -> SimpleRNN(32) -> Dense(1) inference path."""
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 7)
    axis.axis("off")
    axis.set_title("Trained TensorFlow/Keras model: data flow and recurrent cell", fontsize=15, pad=15)

    add_box(axis, 0.25, 4.6, 2.5, 1.35, "24-hour input", "24 steps x 7 features\n(batch, 24, 7)")
    add_box(axis, 3.45, 4.6, 3.1, 1.35, "SimpleRNN: 32 units", "tanh; shared weights each hour\n32 x (7 + 32 + 1) = 1,280 params")
    add_box(axis, 7.3, 4.6, 2.0, 1.35, "Dense: 1", "32 -> 1\n33 params")
    add_box(axis, 10.1, 4.6, 3.2, 1.35, "Forecast", "scaled next-hour power\ninverse-scaled to kW")
    add_arrow(axis, (2.75, 5.28), (3.45, 5.28))
    add_arrow(axis, (6.55, 5.28), (7.3, 5.28), "final h_24")
    add_arrow(axis, (9.3, 5.28), (10.1, 5.28))
    axis.text(3.45, 4.15, "Unroll this same cell once per input hour (24 applications)", fontsize=10, weight="bold")

    add_box(axis, 1.35, 2.15, 2.2, 1.15, "Previous state", "h_(t-1), 32 values")
    add_box(axis, 5.0, 2.15, 3.4, 1.15, "One recurrent cell", "h_t = tanh(W_xh x_t\n+ W_hh h_(t-1) + b_h)")
    add_box(axis, 9.95, 2.15, 2.2, 1.15, "New state", "h_t, 32 values")
    add_arrow(axis, (3.55, 2.73), (5.0, 2.73))
    add_arrow(axis, (8.4, 2.73), (9.95, 2.73))
    add_box(axis, 5.8, 0.45, 1.8, 0.8, "Current input", "x_t, 7 values")
    add_arrow(axis, (6.7, 1.25), (6.7, 2.15))
    axis.add_patch(
        FancyArrowPatch(
            (11.0, 2.1),
            (2.3, 2.1),
            connectionstyle="arc3,rad=-0.42",
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#8a3d3d",
        )
    )
    axis.text(6.4, 1.35, "h_t becomes the previous state at the next hour", ha="center", fontsize=9, color="#8a3d3d")
    axis.text(
        0.3,
        0.2,
        "Input order: " + " | ".join(FEATURE_NAMES),
        fontsize=8,
        va="bottom",
    )
    axis.text(10.0, 3.55, "Total trainable parameters: 1,313", fontsize=10, weight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def select_examples(actual_kw: np.ndarray) -> list[tuple[str, int]]:
    """Select test examples near low, median, and high target quantiles."""
    selections = []
    for label, quantile in (("Low demand (15th percentile)", 0.15), ("Typical demand (50th percentile)", 0.50), ("High demand (98th percentile)", 0.98)):
        target_value = float(np.quantile(actual_kw, quantile))
        index = int(np.argmin(np.abs(actual_kw - target_value)))
        selections.append((label, index))
    if len({index for _, index in selections}) != len(selections):
        raise ValueError("Could not select three distinct low, typical, and high demand examples")
    return selections


def save_real_examples_plot(
    inputs: np.ndarray,
    hidden_states: np.ndarray,
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    examples: list[tuple[str, int]],
    output_path: Path,
    target_min: float,
    target_max: float,
) -> None:
    """Plot real 24-hour target histories and all 32 recurrent activations."""
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), gridspec_kw={"width_ratios": [1.1, 1.4]})
    for row, (label, index) in enumerate(examples):
        history_kw = inputs[row, :, 0] * (target_max - target_min) + target_min
        history_hours = np.arange(-23, 1)
        axes[row, 0].plot(history_hours, history_kw, marker=".", linewidth=1.5, label="Observed history")
        axes[row, 0].scatter([1], [actual_kw[row]], marker="o", s=65, label="Actual next hour", zorder=3)
        axes[row, 0].scatter([1], [predicted_kw[row]], marker="x", s=75, label="Model prediction", zorder=4)
        axes[row, 0].set_title(
            f"{label}\nTest sample {index}: actual {actual_kw[row]:.3f} kW, "
            f"predicted {predicted_kw[row]:.3f} kW"
        )
        axes[row, 0].set_xlabel("Relative hour (forecast at +1)")
        axes[row, 0].set_ylabel("Global active power (kW)")
        axes[row, 0].set_xticks([-23, -18, -12, -6, 0, 1])
        axes[row, 0].grid(alpha=0.25)
        if row == 0:
            axes[row, 0].legend(fontsize=8, loc="upper left")

        heatmap = axes[row, 1].imshow(
            hidden_states[row].T,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        axes[row, 1].set_title("32 hidden-unit activations across the 24-hour sequence")
        axes[row, 1].set_xlabel("Input hour (oldest to most recent)")
        axes[row, 1].set_ylabel("Hidden unit")
        axes[row, 1].set_xticks([0, 5, 11, 17, 23], labels=["1", "6", "12", "18", "24"])
    figure.suptitle("Three real held-out examples processed by the trained RNN", fontsize=16)
    figure.subplots_adjust(left=0.08, right=0.91, top=0.92, bottom=0.07, hspace=0.48, wspace=0.15)
    color_axis = figure.add_axes([0.93, 0.18, 0.015, 0.64])
    figure.colorbar(heatmap, cax=color_axis, label="tanh hidden-state value")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_feature_inputs_plot(inputs: np.ndarray, examples: list[tuple[str, int]], output_path: Path) -> None:
    """Plot each example's seven actual model input features over 24 hours."""
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    images = []
    values = np.stack([inputs[row].T for row in range(len(examples))])
    color_min = float(values.min())
    color_max = float(values.max())
    for row, (axis, (label, index)) in enumerate(zip(axes, examples)):
        image = axis.imshow(
            inputs[row].T,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        images.append(image)
        axis.set_title(f"{label} — test sample {index}")
        axis.set_ylabel("Input feature")
        axis.set_yticks(np.arange(len(FEATURE_NAMES)), labels=FEATURE_NAMES, fontsize=8)
    axes[-1].set_xlabel("Input hour (oldest to most recent)")
    axes[-1].set_xticks([0, 5, 11, 17, 23], labels=["1", "6", "12", "18", "24"])
    figure.suptitle("Actual seven-feature windows fed into the RNN", fontsize=15)
    figure.subplots_adjust(left=0.16, right=0.90, top=0.92, bottom=0.08, hspace=0.42)
    color_axis = figure.add_axes([0.92, 0.18, 0.015, 0.64])
    figure.colorbar(images[-1], cax=color_axis, label="training-scaled input value")
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def write_examples_csv(
    examples: list[tuple[str, int]],
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    output_path: Path,
) -> None:
    """Write the three selected held-out examples and their forecast errors."""
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "test_sample_index", "actual_kW", "predicted_kW", "error_actual_minus_predicted_kW"])
        for row, (label, index) in enumerate(examples):
            writer.writerow(
                [
                    label,
                    index,
                    actual_kw[row],
                    predicted_kw[row],
                    actual_kw[row] - predicted_kw[row],
                ]
            )


def write_report(
    examples: list[tuple[str, int]],
    actual_kw: np.ndarray,
    predicted_kw: np.ndarray,
    output_path: Path,
) -> None:
    """Write a presentation guide with measured outputs for the selected cases."""
    rows = [
        "# Trained RNN: Architecture and Real Test Examples",
        "",
        "These examples use the saved TensorFlow/Keras checkpoint and three actual held-out test sequences. "
        "Cases are selected near the 15th, 50th, and 98th percentiles of actual next-hour demand; "
        "they are not synthetic inputs.",
        "",
        "## Model architecture",
        "",
        "```text",
        "Input: 24 previous hours x 7 training-scaled features",
        "  -> SimpleRNN(32, tanh): 32 hidden activations at each hour",
        "  -> use final hidden state h_24",
        "  -> Dense(1): next-hour prediction in training-scaled units",
        "  -> inverse min-max transform: prediction in kW",
        "```",
        "",
        "The recurrent layer reuses the same input weights, recurrent weights, and bias at all 24 hours. "
        "The RNN layer has 1,280 parameters; the dense output layer has 33, for 1,313 total.",
        "",
        "## Three real held-out predictions",
        "",
        "| Case | Test sample | Actual (kW) | Predicted (kW) | Error: actual - predicted (kW) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row, (label, index) in enumerate(examples):
        rows.append(
            f"| {label} | {index} | {actual_kw[row]:.4f} | {predicted_kw[row]:.4f} "
            f"| {actual_kw[row] - predicted_kw[row]:+.4f} |"
        )
    rows.extend(
        [
            "",
            "## How to explain the visualizations",
            "",
            "- **Architecture diagram:** each input hour contributes seven measurements. The same SimpleRNN cell "
            "updates the hidden state 24 times; the final state is passed to the Dense output layer.",
            "- **Feature-window heatmap:** shows the seven training-scaled feature values the model actually received "
            "for each selected sequence.",
            "- **Prediction and hidden-state panels:** the left plots show the 24-hour Global_active_power history "
            "and compare actual versus predicted next-hour demand. The right heatmaps show how all 32 hidden units "
            "changed during the sequence.",
            "- **Important distinction:** hidden activations are learned internal representations, not direct "
            "appliance labels or causal explanations.",
            "",
            "## Run again",
            "",
            "```text",
            "python models/rnn/visualize_tensorflow_examples.py",
            "```",
            "",
            "The script writes the architecture figure, the two real-example figures, this report, and a CSV "
            "of the three measured predictions under `results/`.",
        ]
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def generate_visualizations(
    model_path: Path,
    results_dir: Path,
    rscript_path: str | None = None,
) -> dict[str, object]:
    """Load the real Keras checkpoint and generate architecture/example outputs."""
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is required. Install it with `python -m pip install tensorflow`."
        ) from error

    model = tf.keras.models.load_model(model_path)
    if len(model.layers) != 2:
        raise ValueError(f"Expected SimpleRNN + Dense, found {len(model.layers)} layers")
    recurrent_layer, output_layer = model.layers
    if not isinstance(recurrent_layer, tf.keras.layers.SimpleRNN) or not isinstance(
        output_layer, tf.keras.layers.Dense
    ):
        raise TypeError("Expected a Keras SimpleRNN followed by a Dense output layer")
    if recurrent_layer.units != 32 or output_layer.units != 1:
        raise ValueError("Checkpoint architecture does not match the expected 32-unit, one-output model")
    if recurrent_layer.count_params() + output_layer.count_params() != 1313:
        raise ValueError("Unexpected parameter count in saved Keras checkpoint")

    test_path = PROJECT_ROOT / "preprocessing" / "test_sequences.rds"
    metadata_path = PROJECT_ROOT / "preprocessing" / "preprocessing_metadata.rds"
    inputs, targets_scaled = load_rds_sequences(test_path, rscript_path)
    if inputs.shape != (6894, 24, 7) or targets_scaled.shape != (6894,):
        raise ValueError(f"Unexpected test shapes: {inputs.shape}, {targets_scaled.shape}")
    target_min, target_max = load_target_scaling(metadata_path, rscript_path)
    target_scale = target_max - target_min
    actual_kw = targets_scaled * target_scale + target_min
    examples = select_examples(actual_kw)
    indices = [index for _, index in examples]
    selected_inputs = inputs[indices].astype(np.float32)
    predictions_scaled = model.predict(selected_inputs, verbose=0).reshape(-1)
    predictions_kw = predictions_scaled * target_scale + target_min

    recurrent_sequence = tf.keras.layers.SimpleRNN(
        recurrent_layer.units,
        activation=recurrent_layer.activation,
        use_bias=recurrent_layer.use_bias,
        return_sequences=True,
        dtype=recurrent_layer.dtype,
    )
    recurrent_sequence(tf.keras.Input(shape=(24, 7)))
    recurrent_sequence.set_weights(recurrent_layer.get_weights())
    hidden_model = tf.keras.Model(recurrent_sequence.input, recurrent_sequence.output)
    hidden_states = hidden_model.predict(selected_inputs, verbose=0)
    dense_predictions_scaled = output_layer(hidden_states[:, -1, :]).numpy().reshape(-1)
    if not np.allclose(predictions_scaled, dense_predictions_scaled, rtol=1e-5, atol=1e-6):
        raise RuntimeError("Reconstructed recurrent states do not reproduce the saved model outputs")
    if not np.isfinite(predictions_kw).all() or not np.isfinite(hidden_states).all():
        raise FloatingPointError("Non-finite model outputs found")

    results_dir.mkdir(parents=True, exist_ok=True)
    architecture_path = results_dir / "rnn_tensorflow_architecture.png"
    examples_path = results_dir / "rnn_tensorflow_real_examples.png"
    features_path = results_dir / "rnn_tensorflow_real_input_features.png"
    report_path = results_dir / "rnn_tensorflow_real_examples.md"
    csv_path = results_dir / "rnn_tensorflow_real_examples.csv"
    selected_actual_kw = actual_kw[indices]
    save_architecture_plot(architecture_path)
    save_real_examples_plot(
        selected_inputs,
        hidden_states,
        selected_actual_kw,
        predictions_kw,
        examples,
        examples_path,
        target_min,
        target_max,
    )
    save_feature_inputs_plot(selected_inputs, examples, features_path)
    write_examples_csv(examples, selected_actual_kw, predictions_kw, csv_path)
    write_report(examples, selected_actual_kw, predictions_kw, report_path)

    for row, (label, index) in enumerate(examples):
        print(
            f"{label}: test sample {index}, actual={selected_actual_kw[row]:.6f} kW, "
            f"predicted={predictions_kw[row]:.6f} kW, "
            f"error={selected_actual_kw[row] - predictions_kw[row]:+.6f} kW"
        )
    print("RNN parameters:", recurrent_layer.count_params() + output_layer.count_params())
    print("Saved architecture:", architecture_path)
    print("Saved real examples:", examples_path)
    print("Saved real input features:", features_path)
    return {
        "examples": [(label, index) for label, index in examples],
        "architecture": architecture_path,
        "predictions": examples_path,
        "input_features": features_path,
        "report": report_path,
        "csv": csv_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "rnn_tensorflow_best.keras")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--rscript", type=str, default=None)
    args = parser.parse_args()
    generate_visualizations(args.model, args.results_dir, args.rscript)


if __name__ == "__main__":
    main()
