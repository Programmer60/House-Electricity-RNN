# Practical PyTorch Vanilla RNN Baseline

## Architecture and data split

The PyTorch model uses `nn.RNN(input_size=7, hidden_size=32,
batch_first=True, nonlinearity="tanh")`, followed by `nn.Linear(32, 1)`. Only
the final output timestep is used for next-hour prediction.

The original chronological training set was split into:

- Training: first `24882` sequences
- Validation: final `2765` sequences from the training period
- Test: untouched `6894` sequences

No preprocessing or normalization was redone. The final test evaluation was
performed after selecting the lowest-validation-loss checkpoint.

## Configuration

- Batch size: `64`
- Learning rate: `0.001`
- Optimizer: Adam
- Maximum epochs: `30`
- Epochs actually trained: `16`
- Patience: `5`
- Seed: `42`
- Best epoch: `11`
- Best validation loss: `0.00929972`
- Runtime: `28.24` seconds

## Test results

- MAE: `0.39160449` kW
- MSE: `0.26817810` kW^2
- RMSE: `0.51785915` kW
- R^2: `0.49392373`
- Prediction range: `0.08605319` to `3.96608853` kW

The predictions were inverse-transformed using the training-only target scaling
parameters. No test sequence was used for optimization or checkpoint selection.

## Mathematical correspondence

The PyTorch recurrent layer automates the same computation represented manually
in NumPy:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
y_hat = W_hy h_T + b_y
L = mean((y - y_hat)^2)
```

The NumPy implementation explicitly derives BPTT and performs plain SGD:

```text
theta_new = theta_old - eta * grad_theta L
```

The PyTorch implementation uses autograd for the gradients and Adam for the
parameter updates. Adam is a practical optimization choice, not a mathematical
requirement of the RNN.
