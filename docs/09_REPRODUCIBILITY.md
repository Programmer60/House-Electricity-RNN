# Reproducibility Notes

This project uses R for preprocessing and Python for RNN modeling.

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

The original 24-timestep RDS files must not be overwritten when generating alternative sequence lengths. All reported metrics are calculated after inverse transformation to kW using training-only target scaling values.
