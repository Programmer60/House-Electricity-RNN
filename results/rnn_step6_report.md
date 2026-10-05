# Step 6: NumPy Vanilla RNN Training Dynamics

## Controlled experiment

The same validated vanilla RNN was retrained with only the epoch count changed:

- Input: `24 x 7`
- Hidden units: `32`
- Optimizer: plain SGD
- Learning rate: `0.001`
- Seed: `42`
- Sample order: chronological, no shuffling
- Epochs: `20`

The model parameters were initialized from the same seed for this experiment.
The unseen test set was used only after all 20 training epochs.

## Results

- Final training loss: `0.00477321`
- Test MAE: `0.48833551` kW
- Test RMSE: `0.59847853` kW
- Test R^2: `0.32408855`
- Runtime: `349.22` seconds

Prediction distribution:

- Prediction range: `0.52873377` to `4.58073821` kW
- Prediction mean: `1.26285354` kW
- Actual range: `0.16043757` to `5.62683305` kW
- Actual mean: `1.00245562` kW

The prediction band remains substantially narrower than the actual test range.
The comparison table should be read together with the plots: lower training
loss alone does not establish improved generalization.

## Mathematical connection

Each training sequence performs the validated forward pass, single-sample loss,
and BPTT gradient calculation. Plain SGD then applies:

```text
theta_new = theta_old - eta * grad_theta L
```

where:

```text
theta = {W_xh, W_hh, b_h, W_hy, b_y}
```

Twenty epochs provide more repeated parameter updates than the two-epoch
baseline. This experiment tests whether additional optimization reduces the
underfitting observed in the baseline without changing the model architecture.
