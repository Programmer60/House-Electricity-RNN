# Preprocessing Write-up: Household Power Consumption → RNN

## Dataset
UCI "Individual Household Electric Power Consumption" dataset:
minute-level readings of one household's electricity use from
16/12/2006 to 26/11/2010 (~2.07 million rows, 9 columns).
Delimiter is `;`, missing readings are coded as `?`.

## Pipeline (matches `preprocessing_RNN.R` section numbers)

1. **Load** — `data.table::fread` for speed on a 2M-row file;
   `na.strings = "?"` converts missing markers to real `NA`s so
   numeric columns parse correctly instead of becoming text.

2. **Datetime** — `Date` + `Time` are separate text columns; they
   are merged into one `POSIXct` timestamp and sorted, since an
   RNN needs a single ordered time index.

3. **Missing values** — ~1.25% of rows have missing readings, and
   they occur in contiguous blocks (sensor outages), not scattered
   randomly. Linear interpolation (`zoo::na.approx`) is used
   because power draw is a continuous physical signal; any leftover
   NAs at the very start/end (nothing to interpolate from) are
   filled with the nearest known value (`na.locf`).

4. **Resampling to hourly** — minute-level data is noisy and huge.
   Aggregating to hourly means reduces the series from ~2,075,000
   rows to ~34,600 rows, smooths sensor noise, and keeps the
   daily/weekly seasonality the RNN is meant to learn.

5. **Feature/target choice** — `Global_active_power` is the
   forecasting target (standard for this dataset); all 7 numeric
   columns are kept as multivariate input features.

6. **Train/test split** — chronological 80/20 split, **not**
   shuffled. Time series must be split by time: the test set is
   the most recent period, simulating genuine forecasting.

7. **Scaling** — Min-Max scaling to [0,1], fit only on the
   training set's min/max and applied to test data. RNN gates
   (tanh/sigmoid) train much better on small, uniform-scale
   inputs. Fitting the scaler on train only avoids leaking future
   information into training.

8. **Sequence windowing** — RNNs consume 3D tensors of shape
   `(samples, timesteps, features)`. A sliding window of 24 past
   hours is used to predict the next hour's `Global_active_power`
   (`timesteps` is easy to change, e.g. 168 for "past week").

9. **Save** — `.rds` files for train/test sequences and scaling
   parameters, so preprocessing and RNN training are cleanly
   separated steps.

## Talking points for the professor
- Why interpolation over deletion: deleting ~26k rows would break
  time continuity, which sequence models depend on.
- Why hourly aggregation: makes training tractable in a lab
  session while preserving the seasonal structure RNNs exploit.
- Why scale on train-only stats: prevents test-set leakage, a
  common pitfall specifically for time series problems.
- Why the specific 3D array shape: it's a hard requirement of
  `keras`'s `layer_simple_rnn`/`layer_lstm`/`layer_gru`.
  
