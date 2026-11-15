# Reproducibility Notes

This project uses R for preprocessing and Python for RNN modeling.

The project was validated with Python 3.13.5, NumPy 2.3.4, Matplotlib 3.10.7,
PyTorch 2.14.1 (CPU), TensorFlow 2.21.0, and R 4.6.1. Install Python
dependencies with `python -m pip install -r requirements.txt`. The R
preprocessing uses `data.table`, `lubridate`, `zoo`, `dplyr`, and `ggplot2`.

The raw dataset is included at
`individual+household+electric+power+consumption/household_power_consumption.txt`.
To regenerate the primary RDS inputs, run
`Rscript preprocessing/Preprocessing.R` from the project root. Alternative
sequence-length data is regenerated with
`Rscript preprocessing/generate_sequence_lengths.R` from the project root.
Both scripts use the project-relative dataset path.

## Key controls

- Seed: `42`
- Input features: 7
- Locked sequence length: 24
- Hidden units: 32
- Batch size: 64
- Adam learning rate: `0.001`
- Chronological validation
- No shuffling
- Early stopping patience: 5

## Important artifacts

- Locked data: `preprocessing/train_sequences.rds`, `preprocessing/test_sequences.rds`
- Scaling metadata: `preprocessing/preprocessing_metadata.rds`
- PyTorch checkpoint: `models/rnn_pytorch_best.pt`
- TensorFlow/Keras checkpoint: `models/rnn_tensorflow_best.keras`
- Metrics: `results/rnn_pytorch_metrics.json`
- TensorFlow/Keras metrics: `results/rnn_tensorflow_metrics.json`
- Sequence experiment: `results/rnn_sequence_length_metrics.json`
- Final report: `results/final_project_report.md`
- Trained-model visual guide: `results/rnn_tensorflow_real_examples.md`

The original 24-timestep RDS files must not be overwritten when generating alternative sequence lengths. All reported metrics are calculated after inverse transformation to kW using training-only target scaling values.
