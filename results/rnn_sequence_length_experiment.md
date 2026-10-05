# Step 10: Controlled Sequence-Length Experiment

## Methodology and controls

Alternative sequences were generated in R from the hourly preprocessing pipeline;
Python did not slice the existing 24-timestep arrays. The original
`train_sequences.rds` and `test_sequences.rds` remain unchanged.

The original chronological training period was scaled using its training-only
min/max values, then its sequence stream was split chronologically into a
training prefix and validation suffix. The original test period was generated
as a separate sequence stream. This preserves the locked 24-hour protocol and
prevents future observations from entering an input window. Validation windows
retain earlier training-period observations as historical input context. Test
windows intentionally restart at the test partition boundary, matching the
locked baseline convention; preceding-period values are not used as test input
context.

For every length, the sequence rule was `X[i] = rows i:(i+T-1)` and
`y[i] = Global_active_power at row i+T`. The generated counts are recorded
below. All PyTorch controls stayed fixed: 7 features, 32 hidden units, tanh
RNN, Adam, learning rate `0.001`, batch size `64`, seed `42`, no shuffling,
maximum 30 epochs, and patience 5. The test set was used only after selecting
the lowest-validation-loss checkpoint.

## Results

| Sequence Length | Train Sequences | Validation Sequences | Test Sequences | Actual Epochs | Best Epoch | Best Val Loss | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 hours | 24898 | 2767 | 6912 | 10 | 5 | 0.00933950 | 0.43258984 | 0.55052017 | 0.42855690 |
| 12 hours | 24893 | 2766 | 6906 | 19 | 14 | 0.00937167 | 0.40063578 | 0.52605308 | 0.47835477 |
| 24 hours | 24882 | 2765 | 6894 | 16 | 11 | 0.00929972 | 0.39160449 | 0.51785915 | 0.49392373 |
| 48 hours | 24860 | 2763 | 6870 | 11 | 6 | 0.00942813 | 0.40890399 | 0.52680389 | 0.47287067 |

The 24-hour result reproduced the locked RNN-32 baseline within the required
numerical tolerance. Its MAE, RMSE, and R^2 match the locked values.

## Mathematical interpretation

Increasing `T` requires the recurrence

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)
```

to process more transitions: 6, 12, 24, or 48 recurrent steps. During BPTT,
this connects to repeated Jacobian products such as

```text
partial L / partial h_t is controlled by products of J_k^T
```

Longer sequences can expose gradients to more recurrent multiplications and
increase computation, but performance must be determined empirically.

## Interpretation

- Lowest test RMSE among the tested lengths: **T=24**.
- Highest test R^2 among the tested lengths: **T=24**.
- The 24-hour history provides a useful measured reference point, but these
  four runs do not establish that it is universally optimal.
- Sequence length changes the number of input timesteps and therefore runtime,
  memory use, available historical context, and possible gradient-propagation
  distance. It does not automatically improve or degrade generalization.
- These conclusions apply only to T=6, 12, 24, and 48 under this fixed training
  configuration. No other experiments were run.
