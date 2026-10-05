# RNN Mathematics

## Recurrence

For an input vector `x_t` and hidden state `h_t`:

```text
a_t = W_xh x_t + W_hh h_(t-1) + b_h
h_t = tanh(a_t)
```

The final hidden state produces the next-hour prediction:

```text
y_hat = W_hy h_T + b_y
```

For the baseline:

```text
x_t in R^7
h_t in R^32
W_xh in R^(32 x 7)
W_hh in R^(32 x 32)
b_h in R^32
W_hy in R^(1 x 32)
b_y in R
```

The baseline has:

```text
32*7 + 32*32 + 32 + 1*32 + 1 = 1313 parameters
```

## Loss

The NumPy reference implementation uses one-sample half-squared loss:

```text
L = 1/2 (y - y_hat)^2
```

The factor `1/2` cancels the factor 2 from differentiating the square. The practical PyTorch implementation uses mean squared error over each mini-batch.

## Parameter update

The NumPy implementation uses transparent SGD:

```text
theta_new = theta_old - eta * grad_theta L
```

The PyTorch implementation uses automatic differentiation and Adam. Adam is an optimization choice, not a requirement of the RNN equations.
