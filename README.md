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

The included source dataset is the UCI Individual Household Electric Power
Consumption data. R preprocessing requires `data.table`, `lubridate`, `zoo`,
`dplyr`, and `ggplot2`. Python requirements are listed in `requirements.txt`.

## Documentation

- [Data preprocessing](docs/01_DATA_PREPROCESSING.md)
- [RNN mathematics](docs/02_RNN_MATHEMATICS.md)
- [BPTT and gradient analysis](docs/03_BPTT_AND_GRADIENT_ANALYSIS.md)
- [RNN implementation](docs/04_RNN_IMPLEMENTATION.md)
- [Experiments and results](docs/05_EXPERIMENTS_AND_RESULTS.md)
- [Final project report](results/final_project_report.md)
- [Reproducibility](docs/06_REPRODUCIBILITY.md)

## Main implementations

- NumPy reference: `models/rnn/rnn_numpy.py`
- NumPy training: `models/rnn/train_numpy.py`
- PyTorch baseline: `models/rnn/rnn_pytorch.py`
- TensorFlow/Keras baseline: `models/rnn/rnn_tensorflow.py`
- Real-model architecture and example visualizations: `models/rnn/visualize_tensorflow_examples.py`
- R-generated sequence lengths: `preprocessing/generate_sequence_lengths.R`

## Run the project

Use Python 3.13 or a compatible Python version and install the listed
packages:

```text
python -m pip install -r requirements.txt
```

Install R and the packages listed at the top of
`preprocessing/Preprocessing.R` if you need to regenerate the prepared RDS
files. From the project root, run:

```text
Rscript preprocessing/Preprocessing.R
python models/rnn/rnn_pytorch.py
python models/rnn/rnn_tensorflow.py
python models/rnn/rnn_math_example.py
python models/rnn/visualize_tensorflow_examples.py
```

The preprocessing script writes its RDS outputs to `preprocessing/`, and the
Python runners write model artifacts and reports to `models/` and `results/`.
The included RDS files and trained checkpoints mean the project can also be
reviewed without rerunning training.

## Locked RNN result

The locked practical baseline uses 24 timesteps, 7 features, 32 hidden units, Adam, learning rate `0.001`, batch size `64`, seed `42`, chronological validation, and patience `5`.

On the unseen test set it achieved:

- MAE: `0.39160449 kW`
- RMSE: `0.51785915 kW`
- R2: `0.49392373`

Among the tested sequence lengths 6, 12, 24, and 48 hours, the 24-hour context produced the best measured test RMSE and R2. This is an experimental result, not a universal claim.

## Reproducibility

The project records configuration, histories, checkpoints, metrics, and plots under `results/` and `models/`. The original 24-timestep RDS files remain the locked baseline data interface.
