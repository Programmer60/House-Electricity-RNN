# Data Preprocessing

## R-only preprocessing

The preprocessing pipeline is implemented in `preprocessing/Preprocessing.R`. Python does not redo cleaning, resampling, scaling, or sequence construction.

The raw file uses semicolon-separated columns and `?` for missing values. R merges `Date` and `Time`, sorts chronologically, interpolates missing numeric readings, fills boundary gaps, and aggregates minute readings into hourly means.

## Features and target

The seven input features are:

```text
Global_active_power
Global_reactive_power
Voltage
Global_intensity
Sub_metering_1
Sub_metering_2
Sub_metering_3
```

The target is `Global_active_power`, measured in kW.

## Chronological split and scaling

The hourly data is split chronologically into an earlier 80% training period and a later 20% test period. No random shuffle is used.

Min-max values are learned only from the training period:

```text
scaled = (value - train_min) / (train_max - train_min)
```

The same training min/max values are applied to validation and test data. Predictions are inverse-transformed to kW before reporting MAE, MSE, RMSE, or R2.

## Sequence alignment

For sequence length `T`:

```text
X[i] = observations i through i+T-1
y[i] = Global_active_power at i+T
```

Thus, a 24-hour window predicts the following hour. The original validated arrays are `(27647, 24, 7)` and `(6894, 24, 7)` for train and test.

Alternative lengths were generated in R by `preprocessing/generate_sequence_lengths.R`; the locked 24-timestep files were not overwritten.
