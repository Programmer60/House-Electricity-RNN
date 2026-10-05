# NumPy Vanilla RNN Training

This document describes the Step 4 training loop implemented in
`models/rnn/rnn_numpy.py` and `models/rnn/train_numpy.py`.

## Forward pass

For each input sequence, the RNN starts with a zero hidden state and processes
one input vector at a time:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

The same `W_xh`, `W_hh`, and `b_h` are reused at every timestep. After the
last timestep, the linear output layer produces:

```text
y_hat = W_hy h_T + b_y
```

For this project, one sequence has 24 timesteps and 7 features, and the hidden
state has 32 values.

## Loss and BPTT

The first implementation uses one-sample half-squared error:

```text
L = 1/2 (y - y_hat)^2
```

The factor `1/2` cancels the factor `2` produced by differentiating the square,
so `dL/dy_hat = y_hat - y`.

The output gradients are:

```text
dL/dW_hy = (y_hat - y) h_T^T
dL/db_y  = y_hat - y
dL/dh_T  = W_hy^T (y_hat - y)
```

BPTT then visits the unrolled hidden states in reverse order, from Python
index `23` to `0`. At each timestep:

```text
dL/da_t = dL/dh_t * (1 - h_t^2)
dL/dW_xh += dL/da_t x_t^T
dL/dW_hh += dL/da_t h_(t-1)^T
dL/db_h  += dL/da_t
dL/dh_(t-1) = W_hh^T dL/da_t
```

`W_hh` is shared by every timestep, so its gradient is accumulated across all
24 contributions before the update.

## Gradient descent

After one sequence has produced its BPTT gradients, plain stochastic gradient
descent updates every parameter:

```text
theta_new = theta_old - learning_rate * gradient
```

The baseline learning rate is `0.001`. Training consumes the existing training
sequences in chronological order without shuffling. Each epoch records the
mean of the per-sequence half-squared losses.

## Baseline run

The initial baseline configuration is:

```text
hidden_units = 32
timesteps = 24
features = 7
learning_rate = 0.001
epochs = 2
optimizer = plain stochastic gradient descent
loss_function = 0.5 * (y - y_hat)^2 per sample
random_seed = 42
shuffle = false
```

The full training data has shape `X_train = (27647, 24, 7)` and
`y_train = (27647,)`. The two-epoch run decreased training loss from
`0.0101326` to `0.0094599` in approximately `33.15` seconds, with no NaN or
Inf values.

The separate overfit check used 12 samples and a debugging-only learning rate
of `0.01` for 500 epochs. It reduced loss from `0.0508207` to `0.0034701`.
This higher rate is not the full-data baseline and is not a hyperparameter
sweep.
