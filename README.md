# Household Electricity Consumption Forecasting

This project forecasts next-hour household electricity consumption from the UCI Individual Household Electric Power Consumption dataset.

## Pipeline

```text
Raw minute-level data
  -> R preprocessing
  -> hourly aggregation
  -> train-only min-max scaling
  -> chronological sequence generation
  -> NumPy mathematical RNN
  -> PyTorch practical RNN
  -> TensorFlow/Keras practical RNN
  -> evaluation and controlled experiments
```

The target is `Global_active_power`, reported in kW. Preprocessing remains in R; Python is used for the RNN implementations.

## Documentation

- [Project overview](docs/01_PROJECT_OVERVIEW.md)
- [Data preprocessing](docs/02_DATA_PREPROCESSING.md)
- [RNN mathematics](docs/03_RNN_MATHEMATICS.md)
- [BPTT and gradient analysis](docs/04_BPTT_AND_GRADIENT_ANALYSIS.md)
- [RNN implementation](docs/05_RNN_IMPLEMENTATION.md)
- [Experiments and results](docs/06_EXPERIMENTS_AND_RESULTS.md)
- [RNN versus LSTM](docs/07_RNN_VS_LSTM.md)
- [Viva and presentation notes](docs/08_VIVA_AND_PRESENTATION.md)

## Main implementations

- NumPy reference: `models/rnn/rnn_numpy.py`
- NumPy training: `models/rnn/train_numpy.py`
- PyTorch baseline: `models/rnn/rnn_pytorch.py`
- TensorFlow/Keras baseline: `models/rnn/rnn_tensorflow.py`
- R-generated sequence lengths: `preprocessing/generate_sequence_lengths.R`

## Locked RNN result

The locked practical baseline uses 24 timesteps, 7 features, 32 hidden units, Adam, learning rate `0.001`, batch size `64`, seed `42`, chronological validation, and patience `5`.

On the unseen test set it achieved:

- MAE: `0.39160449 kW`
- RMSE: `0.51785915 kW`
- R2: `0.49392373`

Among the tested sequence lengths 6, 12, 24, and 48 hours, the 24-hour context produced the best measured test RMSE and R2. This is an experimental result, not a universal claim.

## Reproducibility

The project records configuration, histories, checkpoints, metrics, and plots under `results/` and `models/`. The original 24-timestep RDS files remain the locked baseline data interface.
