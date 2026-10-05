"""Forward propagation, loss, and BPTT for the vanilla NumPy RNN.

This module deliberately implements the mathematics directly. It does not
contain an optimizer, batch training, or an automatic differentiation system.
"""

from __future__ import annotations

import numpy as np


class VanillaRNN:
    """A many-to-one vanilla RNN with a linear output layer.

    For one input sequence, the recurrence is:

        h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)
        y_hat = W_hy @ h_T + b_y

    Inputs use shape (timesteps, input_size). The default baseline is
    24 timesteps, 7 features, and 32 hidden units, but the class does not
    hard-code the sequence length.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 1,
        seed: int = 42,
    ) -> None:
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("input_size, hidden_size, and output_size must be positive")

        rng = np.random.default_rng(seed)

        # W_xh maps each input feature vector R^input_size to R^hidden_size.
        self.W_xh = rng.normal(0.0, 0.01, size=(hidden_size, input_size))
        # W_hh carries information from h_(t-1) in R^hidden_size to h_t.
        self.W_hh = rng.normal(0.0, 0.01, size=(hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
        # W_hy maps the final hidden state to the output vector.
        self.W_hy = rng.normal(0.0, 0.01, size=(output_size, hidden_size))
        self.b_y = np.zeros(output_size)

    @property
    def parameter_count(self) -> int:
        """Return the total number of scalar trainable parameters."""
        return sum(parameter.size for parameter in self.parameters())

    def parameters(self) -> tuple[np.ndarray, ...]:
        """Return parameters in the same groups used by the equations."""
        return self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y

    def forward(
        self,
        sequence: np.ndarray,
        initial_hidden: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute one prediction by unrolling the recurrence through time.

        Args:
            sequence: Input with shape (timesteps, input_size).
            initial_hidden: Optional h_0 with shape (hidden_size,). Defaults
                to the zero vector.

        Returns:
            A tuple of (prediction, final_hidden, all_hidden_states), with
            shapes (output_size,), (hidden_size,), and
            (timesteps, hidden_size).
        """
        sequence = np.asarray(sequence, dtype=float)
        if sequence.ndim != 2 or sequence.shape[1] != self.W_xh.shape[1]:
            expected = f"(timesteps, {self.W_xh.shape[1]})"
            raise ValueError(f"sequence must have shape {expected}; got {sequence.shape}")

        if initial_hidden is None:
            hidden = np.zeros(self.W_hh.shape[0])
        else:
            hidden = np.asarray(initial_hidden, dtype=float)
            if hidden.shape != (self.W_hh.shape[0],):
                raise ValueError(
                    f"initial_hidden must have shape ({self.W_hh.shape[0]},); got {hidden.shape}"
                )

        hidden_states = np.empty((sequence.shape[0], self.W_hh.shape[0]))
        for timestep, input_vector in enumerate(sequence):
            # This is h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h).
            hidden = np.tanh(
                self.W_xh @ input_vector + self.W_hh @ hidden + self.b_h
            )
            hidden_states[timestep] = hidden

        # This is y_hat = W_hy h_T + b_y; output activation is linear.
        prediction = self.W_hy @ hidden + self.b_y
        return prediction, hidden, hidden_states

    @staticmethod
    def loss(target: np.ndarray, prediction: np.ndarray) -> float:
        """Return one-sample L = 1/2 * sum((y - y_hat)^2)."""
        target = np.asarray(target, dtype=float)
        prediction = np.asarray(prediction, dtype=float)
        if target.shape != prediction.shape:
            raise ValueError(
                f"target and prediction must have the same shape; got "
                f"{target.shape} and {prediction.shape}"
            )
        return float(0.5 * np.sum((target - prediction) ** 2))

    def backward(
        self,
        sequence: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute one-sample gradients by manually applying BPTT.

        The forward pass is conceptually unrolled as:

            x_0 -> h_0, x_1 -> h_1, ..., x_T -> h_T -> y_hat

        Here ``h_0`` in the recurrence notation is the zero initial state;
        the Python hidden-state array stores the first computed state at index
        zero. The loop therefore visits indexes ``T - 1`` through ``0``.
        """
        target = np.asarray(target, dtype=float)
        prediction, hidden_final, hidden_states = self.forward(sequence)
        if target.shape != prediction.shape:
            raise ValueError(
                f"target must have shape {prediction.shape}; got {target.shape}"
            )

        # Output layer: dL/dy_hat = y_hat - y.
        dL_dyhat = prediction - target
        dL_dWhy = np.outer(dL_dyhat, hidden_final)
        dL_db_y = dL_dyhat.copy()

        # The output gradient is the starting gradient for the final hidden state.
        dL_dh = self.W_hy.T @ dL_dyhat
        dL_dWxh = np.zeros_like(self.W_xh)
        dL_dWhh = np.zeros_like(self.W_hh)
        dL_dbh = np.zeros_like(self.b_h)

        sequence = np.asarray(sequence, dtype=float)
        zero_hidden = np.zeros(self.W_hh.shape[0])
        for timestep in range(sequence.shape[0] - 1, -1, -1):
            hidden = hidden_states[timestep]

            # tanh'(a_t) = 1 - tanh(a_t)^2 = 1 - h_t^2.
            dL_da = dL_dh * (1.0 - hidden**2)
            dL_dWxh += np.outer(dL_da, sequence[timestep])
            previous_hidden = (
                hidden_states[timestep - 1] if timestep > 0 else zero_hidden
            )
            # W_hh is reused at every timestep, so these terms accumulate.
            dL_dWhh += np.outer(dL_da, previous_hidden)
            dL_dbh += dL_da

            # Pass the recurrent gradient to h_(t-1).
            dL_dh = self.W_hh.T @ dL_da

        gradients = {
            "W_xh": dL_dWxh,
            "W_hh": dL_dWhh,
            "b_h": dL_dbh,
            "W_hy": dL_dWhy,
            "b_y": dL_db_y,
        }
        expected_shapes = {
            "W_xh": self.W_xh.shape,
            "W_hh": self.W_hh.shape,
            "b_h": self.b_h.shape,
            "W_hy": self.W_hy.shape,
            "b_y": self.b_y.shape,
        }
        for name, gradient in gradients.items():
            if gradient.shape != expected_shapes[name]:
                raise AssertionError(
                    f"gradient {name} has shape {gradient.shape}; "
                    f"expected {expected_shapes[name]}"
                )
        return gradients

    def numerical_gradient(
        self,
        parameter_name: str,
        index: tuple[int, ...],
        sequence: np.ndarray,
        target: np.ndarray,
        epsilon: float = 1e-5,
    ) -> float:
        """Approximate one scalar gradient with a central finite difference."""
        parameter = getattr(self, parameter_name)
        original_value = parameter[index]
        parameter[index] = original_value + epsilon
        loss_plus = self.loss(target, self.forward(sequence)[0])
        parameter[index] = original_value - epsilon
        loss_minus = self.loss(target, self.forward(sequence)[0])
        parameter[index] = original_value
        return (loss_plus - loss_minus) / (2.0 * epsilon)

    def gradient_check(
        self,
        sequence: np.ndarray,
        target: np.ndarray,
        epsilon: float = 1e-5,
    ) -> dict[str, dict[str, float]]:
        """Compare analytical and numerical gradients for five parameters."""
        analytical = self.backward(sequence, target)
        selected = {
            "W_xh": (0, 0),
            "W_hh": (0, 0),
            "b_h": (0,),
            "W_hy": (0, 0),
            "b_y": (0,),
        }
        checks: dict[str, dict[str, float]] = {}
        for parameter_name, index in selected.items():
            analytical_value = float(analytical[parameter_name][index])
            numerical_value = self.numerical_gradient(
                parameter_name, index, sequence, target, epsilon
            )
            relative_error = abs(analytical_value - numerical_value) / max(
                1.0, abs(analytical_value), abs(numerical_value)
            )
            checks[parameter_name] = {
                "analytical": analytical_value,
                "numerical": numerical_value,
                "relative_error": relative_error,
            }
        return checks

    def apply_gradients(
        self,
        gradients: dict[str, np.ndarray],
        learning_rate: float = 0.001,
    ) -> None:
        """Apply plain SGD: theta <- theta - learning_rate * gradient."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        for name, parameter in zip(
            ("W_xh", "W_hh", "b_h", "W_hy", "b_y"), self.parameters()
        ):
            gradient = gradients[name]
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"gradient {name} has shape {gradient.shape}; "
                    f"expected {parameter.shape}"
                )
            parameter -= learning_rate * gradient

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        learning_rate: float = 0.001,
        print_progress: bool = True,
    ) -> list[float]:
        """Train sequentially with one-sample plain SGD updates.

        Samples are consumed in their existing order. No shuffling or test
        data is involved. The returned loss is the mean of the one-sample
        half-squared losses after each complete epoch.
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if X_train.ndim != 3:
            raise ValueError("X_train must have shape (samples, timesteps, features)")
        if y_train.shape != (X_train.shape[0], self.W_hy.shape[0]):
            if self.W_hy.shape[0] == 1 and y_train.shape == (X_train.shape[0],):
                y_train = y_train[:, None]
            else:
                raise ValueError(
                    f"y_train must have shape ({X_train.shape[0]}, {self.W_hy.shape[0]})"
                )
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        history: list[float] = []
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for sequence, target in zip(X_train, y_train):
                prediction, _, _ = self.forward(sequence)
                epoch_loss += self.loss(target, prediction)
                gradients = self.backward(sequence, target)
                self.apply_gradients(gradients, learning_rate)

            mean_loss = epoch_loss / X_train.shape[0]
            history.append(mean_loss)
            if print_progress:
                print(f"Epoch {epoch}   Loss: {mean_loss:.8f}")
        return history


def verify_baseline_forward_pass() -> None:
    """Verify baseline dimensions, parameter count, and one small pass."""
    model = VanillaRNN(input_size=7, hidden_size=32, output_size=1, seed=42)
    sample = np.linspace(0.0, 1.0, num=24 * 7, dtype=float).reshape(24, 7)
    prediction, final_hidden, hidden_states = model.forward(sample)

    assert model.W_xh.shape == (32, 7)
    assert model.W_hh.shape == (32, 32)
    assert model.b_h.shape == (32,)
    assert model.W_hy.shape == (1, 32)
    assert model.b_y.shape == (1,)
    assert model.parameter_count == 1313
    assert prediction.shape == (1,)
    assert final_hidden.shape == (32,)
    assert hidden_states.shape == (24, 32)
    assert np.isfinite(prediction).all()

    print("Baseline parameter count:", model.parameter_count)
    print("W_xh shape:", model.W_xh.shape)
    print("W_hh shape:", model.W_hh.shape)
    print("b_h shape:", model.b_h.shape)
    print("W_hy shape:", model.W_hy.shape)
    print("b_y shape:", model.b_y.shape)
    print("Sample prediction:", prediction)
    print("Forward-pass verification: passed")


def verify_bptt() -> None:
    """Run the required synthetic loss, shape, finiteness, and gradient checks."""
    model = VanillaRNN(input_size=2, hidden_size=3, output_size=1, seed=7)
    sequence = np.linspace(-0.4, 0.6, num=4 * 2).reshape(4, 2)
    target = np.array([0.25])
    prediction, _, hidden_states = model.forward(sequence)
    loss = model.loss(target, prediction)
    gradients = model.backward(sequence, target)
    checks = model.gradient_check(sequence, target)

    assert np.isfinite(loss)
    assert hidden_states.shape == (4, 3)
    assert all(gradient.shape == parameter.shape for gradient, parameter in zip(
        gradients.values(), model.parameters()
    ))
    assert all(np.isfinite(gradient).all() for gradient in gradients.values())
    assert all(check["relative_error"] < 1e-6 for check in checks.values())

    print("Synthetic prediction:", prediction)
    print("Synthetic loss:", loss)
    print("Synthetic gradient shapes: passed")
    for parameter_name, check in checks.items():
        print(
            f"{parameter_name}[selected] analytical={check['analytical']:.12g} "
            f"numerical={check['numerical']:.12g} "
            f"relative_error={check['relative_error']:.3e}"
        )
    print("Synthetic BPTT gradient check: passed")


if __name__ == "__main__":
    verify_baseline_forward_pass()
    verify_bptt()
