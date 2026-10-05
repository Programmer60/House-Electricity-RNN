# RNN versus LSTM

## Current status

The vanilla RNN is complete and frozen. LSTM implementation and results belong to another team member and should be added here only after independent validation.

## Comparison framework

Both models should use the same:

- R-generated data and feature ordering
- Chronological train/validation/test policy
- Training-only scaling parameters
- Target and kW inverse transformation
- Evaluation metrics: MAE, MSE, RMSE, and R2
- Test-set isolation

## Mathematical distinction

The vanilla RNN carries one hidden state:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

An LSTM adds a cell state and gates that regulate information flow. Its additional parameters change both capacity and optimization behavior.

## Scientific rule

Do not claim that LSTM is better merely because it is more complex. Compare measured validation and unseen-test results under matched data and evaluation conditions. The RNN baseline here is the reference model, not a claim that it is universally inferior.
