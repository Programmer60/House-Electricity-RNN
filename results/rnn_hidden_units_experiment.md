# Step 9: Controlled Hidden-Unit Experiment

## Experimental control

Only the hidden dimension changed across RNN-16, RNN-32, and RNN-64. All runs
used the same input features, 24 timesteps, chronological data split, Adam
optimizer, learning rate `0.001`, batch size `64`, seed `42`, no shuffling,
maximum 30 epochs, and patience 5. The test set was not used for checkpoint
selection.

The split was identical for every run: `24,882` training sequences, `2,765`
validation sequences, and `6,894` untouched test sequences. Reusing seed `42`
controls randomization, but different hidden sizes necessarily create different
parameter tensor dimensions, so initial parameter arrays are not identical
element-for-element.

## Results

| Model | Hidden Units | Best Epoch | Best Val Loss | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---|---:|---:|---:|---:|---:|---:|
| RNN-16 | 16 | 5 | 0.00926456 | 0.42159388 | 0.54135555 | 0.44695835 |
| RNN-32 | 32 | 11 | 0.00929972 | 0.39160449 | 0.51785915 | 0.49392373 |
| RNN-64 | 64 | 8 | 0.00942270 | 0.41660628 | 0.53069077 | 0.46853370 |

## Parameter counts

For input dimension `d=7`, hidden dimension `H`, and output dimension `1`:

```text
H*d + H^2 + H + H*1 + 1 = H(7) + H^2 + 2H + 1
```

| Hidden Units | Parameters |
|---:|---:|
| 16 | 401 |
| 32 | 1313 |
| 64 | 4673 |

The recurrent matrix `W_hh` alone grows from `H x H`, so increasing hidden
units raises both representational capacity and computational/storage cost.

## Interpretation

- Validation loss was lowest for RNN-16 among the tested configurations.
- Test RMSE was lowest for RNN-32 among the tested configurations.
- Test R^2 was highest for RNN-32 among the tested configurations.
- RNN-32 reproduced the locked baseline within the audit tolerance: recorded MAE `0.39160449` versus experiment `0.39160449`, recorded RMSE `0.51785915` versus experiment `0.51785915`, and recorded R^2 `0.49392373` versus experiment `0.49392373`.
- Training and validation curves should be considered together when assessing overfitting; lower training loss alone does not establish better generalization.

These conclusions apply only to the three tested hidden dimensions and this
fixed training configuration. They do not establish a universal optimum.
