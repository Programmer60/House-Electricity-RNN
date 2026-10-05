# Experiments and Results

## NumPy training dynamics

The NumPy model's full-data training loss decreased from `0.0101326` after the first epoch to `0.0047732` after 20 epochs. Test RMSE improved from `0.8105 kW` at 2 epochs to `0.5985 kW` at 20 epochs.

## Hidden units

The controlled PyTorch hidden-unit experiment tested 16, 32, and 64 units with all other controls fixed.

| Model | Parameters | Test RMSE (kW) | Test R2 |
|---|---:|---:|---:|
| RNN-16 | 401 | 0.5414 | 0.4470 |
| RNN-32 | 1313 | 0.5179 | 0.4939 |
| RNN-64 | 4673 | 0.5307 | 0.4685 |

RNN-32 performed best on the measured test metrics. This is not a universal optimum claim.

## Sequence length

The R-controlled experiment tested 6, 12, 24, and 48 hours.

| Length | Test RMSE (kW) | Test R2 |
|---:|---:|---:|
| 6 | 0.5505 | 0.4286 |
| 12 | 0.5261 | 0.4784 |
| 24 | 0.5179 | 0.4939 |
| 48 | 0.5268 | 0.4729 |

Among the tested contexts, 24 hours produced the lowest RMSE and highest R2. Increasing context to 48 hours did not improve generalization.

## Interpretation

These are controlled results for the tested configurations. Training loss, validation loss, test metrics, prediction plots, and error plots must be considered together.

Detailed records are in the `results/` directory.
