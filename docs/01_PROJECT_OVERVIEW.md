# Project Overview

## Objective

Forecast the next hourly value of `Global_active_power` for one household using recurrent neural networks.

## Dataset

The project uses the UCI Individual Household Electric Power Consumption dataset. It contains minute-level readings from December 2006 through November 2010.

## Team boundary

Member 1 owns the vanilla RNN work:

- NumPy forward propagation and BPTT
- Mathematical gradient analysis
- SGD training
- PyTorch vanilla RNN
- RNN experiments, evaluation, and report material

LSTM work belongs to another team member. Preprocessing and shared evaluation integration remain in R.

## Development phases

1. Inspect and validate R preprocessing outputs.
2. Implement the NumPy forward pass.
3. Implement loss, BPTT, and numerical gradient checking.
4. Train the NumPy reference model.
5. Evaluate the unseen test set.
6. Analyze vanishing gradients.
7. Implement the practical PyTorch RNN.
8. Run controlled hidden-unit and sequence-length experiments.
9. Freeze the vanilla RNN baseline.

## Scientific rule

Training behavior, validation behavior, unseen-test metrics, and plots must be interpreted together. Lower training loss alone does not prove better generalization.
