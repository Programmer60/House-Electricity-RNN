# RNN Implementation

## NumPy reference implementation

`models/rnn/rnn_numpy.py` exposes the mathematical reference implementation:

- Explicit timestep-by-timestep forward propagation
- One-sample loss
- Output-layer gradients
- Manual BPTT
- Numerical gradient checking
- Plain SGD

Analytical gradients were checked against central finite differences with relative errors around `1e-12`.

## PyTorch practical implementation

`models/rnn/rnn_pytorch.py` uses:

```text
nn.RNN(input_size=7, hidden_size=32,
       batch_first=True, nonlinearity="tanh")
```

followed by `nn.Linear(32, 1)`. Only the final output timestep is used. Mini-batches use shape `(batch, 24, 7)` and targets use shape `(batch, 1)`.

Adam with learning rate `0.001` provides practical optimization efficiency. PyTorch autograd computes gradients automatically; this does not replace the NumPy mathematical implementation.

## Locked practical result

The locked PyTorch baseline used chronological validation, patience 5, seed 42, batch size 64, and no shuffling. It selected epoch 11 and achieved:

- MAE: `0.39160449 kW`
- RMSE: `0.51785915 kW`
- R2: `0.49392373`

The checkpoint is `models/rnn_pytorch_best.pt`.
