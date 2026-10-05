"""Step 7: theoretical and empirical gradient-dynamics analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .evaluate_numpy import load_saved_model
    from .train_numpy import load_rds_sequences
except ImportError:
    from evaluate_numpy import load_saved_model
    from train_numpy import load_rds_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def propagate_hidden_gradients(
    model,
    sequence: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ||dL/dh_t||, ||dL/da_t||, and hidden states for t=23,...,0."""
    prediction, _, hidden_states = model.forward(sequence)
    dL_dyhat = prediction - np.asarray(target, dtype=float)
    dL_dh = model.W_hy.T @ dL_dyhat
    hidden_gradient_norms = []
    preactivation_gradient_norms = []

    for timestep in range(sequence.shape[0] - 1, -1, -1):
        hidden = hidden_states[timestep]
        hidden_gradient_norms.append(float(np.linalg.norm(dL_dh)))
        dL_da = dL_dh * (1.0 - hidden**2)
        preactivation_gradient_norms.append(float(np.linalg.norm(dL_da)))
        dL_dh = model.W_hh.T @ dL_da

    return (
        np.asarray(hidden_gradient_norms),
        np.asarray(preactivation_gradient_norms),
        hidden_states,
    )


def save_gradient_plot(
    timesteps: np.ndarray,
    hidden_gradient_norms: np.ndarray,
    preactivation_gradient_norms: np.ndarray,
    output_path: Path,
) -> None:
    """Plot gradient magnitude on a base-10 logarithmic scale."""
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(timesteps, np.log10(hidden_gradient_norms), marker="o", label=r"$||dL/dh_t||$")
    axis.plot(
        timesteps,
        np.log10(preactivation_gradient_norms),
        marker="x",
        label=r"$||dL/da_t||$",
    )
    axis.set_xlabel("Timestep t (backward order: 23 to 0)")
    axis.set_ylabel("log10 gradient norm")
    axis.set_title("Gradient propagation through the trained vanilla RNN")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_analysis_document(
    diagnostics: dict[str, object],
    output_path: Path,
) -> None:
    """Write the mathematical derivation and experiment interpretation."""
    singular_values = diagnostics["W_hh"]["singular_values"]
    hidden_norms = diagnostics["gradient_propagation"]["dL_dh_norms"]
    preactivation_norms = diagnostics["gradient_propagation"]["dL_da_norms"]
    interpretation = diagnostics["interpretation"]
    output_path.write_text(
        f"""# Vanilla RNN Vanishing and Exploding Gradient Analysis

## 1. RNN recurrence

The validated vanilla RNN uses column vectors and the equations:

```text
a_t = W_xh x_t + W_hh h_(t-1) + b_h
h_t = tanh(a_t)
y_hat = W_hy h_T + b_y
```

The recurrent term `W_hh h_(t-1)` makes the current hidden state depend on
the previous hidden state. The same `W_hh` is reused at every timestep.

## 2. Jacobian convention and recurrent gradient

All hidden states and their gradients are treated as column vectors. A small
change in the previous hidden state gives:

```text
da_(t+1) = W_hh dh_t
```

Let:

```text
D_(t+1) = diag(1 - h_(t+1)^2)
```

because `tanh'(a_(t+1)) = 1 - tanh(a_(t+1))^2`. Therefore:

```text
dh_(t+1) = D_(t+1) W_hh dh_t
```

The Jacobian mapping `h_t` to `h_(t+1)` is thus:

```text
J_(t+1) = D_(t+1) W_hh
```

For column-vector gradients, the chain rule uses the transpose Jacobian:

```text
dL/dh_t = J_(t+1)^T dL/dh_(t+1)
         = W_hh^T D_(t+1) dL/dh_(t+1)
```

This is equivalent to writing the product in the opposite orientation when
using row-vector gradients.

## 3. Repeated matrix multiplication

Moving from timestep `T` back to an earlier timestep repeatedly applies these
Jacobians:

```text
dL/dh_t = J_(t+1)^T J_(t+2)^T ... J_T^T dL/dh_T
```

Equivalently, ignoring orientation and focusing on magnitude:

```text
||dL/dh_t|| is controlled by the product of D_k W_hh terms,
for k = t+1, ..., T.
```

Repeated multiplication can shrink the gradient toward zero or enlarge it
exponentially. This is the mathematical source of the vanishing/exploding
gradient vulnerability.

## 4. Scalar intuition

These simplified values illustrate repeated multiplication; they are not the
exact gradients of this trained model:

```text
0.5^10 = 0.0009765625
0.5^20 = 0.00000095367431640625

1.5^10 = 57.6650390625
1.5^20 = 3325.256730079651
```

An effective derivative below one produces decay, while one above one produces
growth. In the real network, the effective factors are matrix Jacobians and
differ from unit to unit and timestep to timestep.

## 5. Tanh contribution

```text
tanh'(a_t) = 1 - tanh(a_t)^2
```

For finite `a_t`, the derivative satisfies `0 < tanh'(a_t) <= 1`. When a unit
saturates near `tanh(a_t) = 1` or `tanh(a_t) = -1`, the derivative approaches
zero:

```text
1 - tanh(a_t)^2 ~= 0
```

The diagonal matrix `D_t` therefore attenuates the recurrent gradient in
`W_hh^T D_(t+1) dL/dh_(t+1)`. This attenuation can compound across many
timesteps.

## 6. Actual saved-model diagnostics

The analysis used `results/rnn_numpy_model.npz`, the 2-epoch baseline, and the
first real test sequence with shape `(24, 7)`. These diagnostics describe the
recurrent matrix but do not by themselves prove vanishing or exploding
behavior.

- `W_hh` shape: `{diagnostics['W_hh']['shape']}`
- Frobenius norm: `{diagnostics['W_hh']['frobenius_norm']:.8f}`
- Spectral norm: `{diagnostics['W_hh']['spectral_norm']:.8f}`
- Minimum singular value: `{min(singular_values):.8f}`
- Maximum singular value: `{max(singular_values):.8f}`

Singular values:

```text
{', '.join(f'{value:.8f}' for value in singular_values)}
```

## 7. Empirical gradient propagation

The starting gradient was `dL/dh_T = W_hy^T (y_hat - y)`. For each timestep,
the analysis recorded the column-vector norms before applying the local tanh
derivative, then propagated with:

```text
dL/dh_(t-1) = W_hh^T [dL/dh_t * (1 - h_t^2)]
```

The recorded values are listed in backward order, `t = 23, ..., 0`:

| t | ||dL/dh_t|| | ||dL/da_t|| |
|---:|---:|---:|
{''.join(f'| {timestep} | {hidden_norm:.8e} | {preactivation_norm:.8e} |{chr(10)}' for timestep, hidden_norm, preactivation_norm in zip(diagnostics['gradient_propagation']['timesteps'], hidden_norms, preactivation_norms))}

The plot is saved as `results/rnn_gradient_propagation.png` with the vertical
axis `log10(gradient norm)`.

## 8. Interpretation and limitations

{interpretation}

Vanilla RNNs are susceptible to vanishing/exploding gradients because the
recurrent gradient involves repeated multiplication of Jacobian terms. The
severity depends on the recurrent weights, activation derivatives, sequence
length, and learned state dynamics. This one sequence and one saved checkpoint
cannot establish behavior for every sequence or every training stage.
""",
        encoding="utf-8",
    )


def run_analysis(results_dir: Path) -> dict[str, object]:
    """Analyze the saved baseline and create diagnostics and documentation."""
    model = load_saved_model(results_dir / "rnn_numpy_model.npz")
    X_test, y_test = load_rds_sequences(PROJECT_ROOT / "preprocessing" / "test_sequences.rds")
    sequence = X_test[0]
    target = np.asarray([y_test[0]])
    hidden_gradient_norms, preactivation_gradient_norms, hidden_states = propagate_hidden_gradients(
        model, sequence, target
    )
    singular_values = np.linalg.svd(model.W_hh, compute_uv=False)
    timesteps = np.arange(sequence.shape[0] - 1, -1, -1)
    if not np.isfinite(hidden_gradient_norms).all() or not np.isfinite(preactivation_gradient_norms).all():
        raise FloatingPointError("NaN or Inf found in propagated gradient norms")

    hidden_ratio = float(hidden_gradient_norms[-1] / hidden_gradient_norms[0])
    max_hidden_norm = float(hidden_gradient_norms.max())
    min_hidden_norm = float(hidden_gradient_norms.min())
    if max_hidden_norm / min_hidden_norm > 1e6:
        interpretation = (
            "This sequence shows strong variation in gradient magnitude across "
            "the 24 timesteps; the ratio of largest to smallest hidden-state "
            "gradient norm is greater than 1e6. This is evidence of severe "
            "gradient scale variation for this checkpoint and sequence, not a "
            "universal claim about every sequence."
        )
    else:
        interpretation = (
            "This particular checkpoint and sequence do not show severe "
            "vanishing or exploding behavior: the hidden-state gradient norms "
            f"range from {min_hidden_norm:.8e} to {max_hidden_norm:.8e}. "
            "That does not remove the theoretical vulnerability of vanilla RNNs."
        )

    diagnostics = {
        "model": "results/rnn_numpy_model.npz",
        "sequence": {"source": "test_sequences.rds", "index": 0, "shape": list(sequence.shape)},
        "W_hh": {
            "shape": list(model.W_hh.shape),
            "frobenius_norm": float(np.linalg.norm(model.W_hh, ord="fro")),
            "spectral_norm": float(singular_values[0]),
            "singular_values": singular_values.tolist(),
        },
        "gradient_propagation": {
            "timesteps": timesteps.tolist(),
            "dL_dh_norms": hidden_gradient_norms.tolist(),
            "dL_da_norms": preactivation_gradient_norms.tolist(),
            "log10_dL_dh_norms": np.log10(hidden_gradient_norms).tolist(),
            "min_dL_dh_norm": min_hidden_norm,
            "max_dL_dh_norm": max_hidden_norm,
            "earliest_to_latest_norm_ratio": hidden_ratio,
        },
        "interpretation": interpretation,
        "finite": True,
    }
    plot_path = results_dir / "rnn_gradient_propagation.png"
    json_path = results_dir / "rnn_gradient_analysis.json"
    document_path = PROJECT_ROOT / "mathematics" / "rnn_gradient_analysis.md"
    save_gradient_plot(timesteps, hidden_gradient_norms, preactivation_gradient_norms, plot_path)
    json_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    save_analysis_document(diagnostics, document_path)
    print("W_hh Frobenius norm:", diagnostics["W_hh"]["frobenius_norm"])
    print("W_hh spectral norm:", diagnostics["W_hh"]["spectral_norm"])
    print("Hidden gradient norm range:", min_hidden_norm, max_hidden_norm)
    print("Saved gradient plot:", plot_path)
    print("Saved analysis:", document_path)
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    run_analysis(args.results_dir)


if __name__ == "__main__":
    main()
