"""Small numerical RNN example for explaining the mathematics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_example(output_path: Path) -> dict[str, object]:
    """Run a three-step, two-unit RNN with hand-chosen values."""
    inputs = np.array([[0.2], [0.5], [0.1]], dtype=float)
    target = 0.8
    W_xh = np.array([[0.5], [-0.4]])
    W_hh = np.array([[0.3, 0.1], [0.2, 0.4]])
    b_h = np.array([0.1, -0.1])
    W_hy = np.array([[0.6, -0.2]])
    b_y = 0.05

    hidden_states = []
    hidden = np.zeros(2)
    pre_activations = []
    for current_input in inputs:
        pre_activation = W_xh[:, 0] * current_input[0] + W_hh @ hidden + b_h
        hidden = np.tanh(pre_activation)
        pre_activations.append(pre_activation.copy())
        hidden_states.append(hidden.copy())
    hidden_states_array = np.array(hidden_states)
    prediction = float((W_hy @ hidden_states_array[-1])[0] + b_y)
    loss = 0.5 * (target - prediction) ** 2
    error_gradient = prediction - target

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.4, 1]})
    timesteps = np.arange(1, len(inputs) + 1)
    axes[0].plot(timesteps, hidden_states_array[:, 0], "o-", label="h1")
    axes[0].plot(timesteps, hidden_states_array[:, 1], "o-", label="h2")
    axes[0].set_xticks(timesteps)
    axes[0].set_title("Tiny RNN unrolled through three timesteps")
    axes[0].set_xlabel("timestep")
    axes[0].set_ylabel("hidden state value")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].axis("off")
    axes[1].text(
        0.01,
        0.95,
        "Forward pass:  h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)\n"
        f"Inputs: x = {[float(value[0]) for value in inputs]}\n"
        f"Final hidden state: h_3 = {np.round(hidden_states_array[-1], 4).tolist()}\n"
        f"Prediction: y_hat = 0.6 h1 - 0.2 h2 + 0.05 = {prediction:.4f}\n"
        f"Target: y = {target:.4f}    Loss: 1/2(y - y_hat)^2 = {loss:.4f}\n"
        f"Output error gradient: dL/dy_hat = y_hat - y = {error_gradient:.4f}",
        va="top",
        family="monospace",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return {
        "inputs": inputs[:, 0].tolist(),
        "pre_activations": np.round(pre_activations, 8).tolist(),
        "hidden_states": np.round(hidden_states_array, 8).tolist(),
        "prediction": prediction,
        "target": target,
        "loss": loss,
        "output_error_gradient": error_gradient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "rnn_math_example.png",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run_example(args.output)
    print("Small RNN mathematics example")
    for timestep, (input_value, pre_activation, hidden) in enumerate(
        zip(result["inputs"], result["pre_activations"], result["hidden_states"]),
        start=1,
    ):
        print(f"t={timestep}: x={input_value:.1f}, a={pre_activation}, h={hidden}")
    print(f"y_hat={result['prediction']:.8f}")
    print(f"loss={result['loss']:.8f}")
    print(f"dL/dy_hat={result['output_error_gradient']:.8f}")
    print(f"Saved visualization: {args.output}")


if __name__ == "__main__":
    main()
