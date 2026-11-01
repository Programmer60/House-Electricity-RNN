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

## Small numerical example for presentation

The script `models/rnn/rnn_math_example.py` uses only one input feature and
two hidden units so every calculation can be shown on a slide. It processes
three input values:

```text
x_1 = 0.2, x_2 = 0.5, x_3 = 0.1
h_0 = [0, 0]^T
```

The hand-chosen parameters are:

```text
W_xh = [[ 0.5], [-0.4]]       W_hh = [[0.3, 0.1], [0.2, 0.4]]
b_h  = [0.1, -0.1]^T          W_hy = [0.6, -0.2]
b_y  = 0.05
```

At the first timestep:

```text
a_1 = W_xh x_1 + W_hh h_0 + b_h
    = [0.2, -0.18]^T
h_1 = tanh(a_1)
    = [0.1974, -0.1781]^T
```

Repeating the same recurrence gives:

```text
h_2 = [0.3726, -0.3201]^T
h_3 = [0.2258, -0.1911]^T
```

The output layer then gives:

```text
y_hat = W_hy h_3 + b_y
      = 0.6(0.2258) - 0.2(-0.1911) + 0.05
      = 0.2237
```

For target `y = 0.8`, the half-squared loss is:

```text
L = 1/2(y - y_hat)^2 = 0.1661
dL/dy_hat = y_hat - y = -0.5763
```

This makes the key idea visible: each new hidden state depends on both the
current input and the previous hidden state. Run the script to print all
values and create `results/rnn_math_example.png`:

```text
python models/rnn/rnn_math_example.py
```
