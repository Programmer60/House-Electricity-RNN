# Viva and Presentation Notes

## Explain the pipeline

Raw minute readings are cleaned and aggregated in R. The resulting hourly data is scaled with training-only min/max values, converted into chronological sliding windows, and passed to a vanilla RNN.

## Explain one prediction

Each input `x_t` updates the hidden state using the tanh recurrence. After the final input hour, the linear output layer maps `h_T` to the next-hour prediction `y_hat`.

## Show the mathematics with a small example

Use `models/rnn/rnn_math_example.py` before the presentation. It creates a
three-timestep visualization with two hidden units, which is much easier to
explain than the production shape `(24, 7)` and 32 hidden units.

Suggested explanation:

1. Start with `h_0 = [0, 0]`.
2. Feed `x_1 = 0.2`; calculate `a_1`, then apply `tanh` to obtain `h_1`.
3. Feed `x_2` and `x_3`; the previous hidden state is carried forward each time.
4. Use only `h_3` in the output equation to calculate `y_hat`.
5. Compare `y_hat` with `y = 0.8` using the half-squared loss.
6. Explain that BPTT sends the output error backward through `h_3`, `h_2`,
   and `h_1`, while accumulating gradients for the shared `W_hh`.

The generated figure is `results/rnn_math_example.png`. The exact numerical
values and equations are documented in `docs/03_RNN_MATHEMATICS.md`.

## Explain BPTT

The RNN is unrolled through time. Since the same recurrent matrix is reused, its gradient accumulates contributions from every timestep. The chain rule sends gradients backward through the recurrent Jacobians.

## Explain the NumPy/PyTorch distinction

NumPy makes the equations and BPTT explicit and verifies gradients numerically. PyTorch uses the same architecture but relies on autograd and Adam for efficient mini-batch training.

## Explain the gradient result

For the analyzed checkpoint and sequence, the hidden gradient norm fell from roughly `4.97e-4` to `2.40e-32`. This is an empirical example of vanishing behavior, not a claim that every RNN always has severe vanishing gradients.

## Explain the experiments

The hidden-unit experiment found RNN-32 best among 16, 32, and 64 units. The sequence-length experiment found 24 hours best among 6, 12, 24, and 48 hours. These are measured results for the controlled configurations only.

## Likely questions

**Why not shuffle the time series?**  Shuffling would weaken the chronological forecasting setup and can cause leakage across temporal structure.

**Why scale using training values only?**  Using future test statistics would leak information into the model evaluation.

**Why use Adam in PyTorch but SGD in NumPy?**  SGD keeps the mathematical reference transparent; Adam is a practical optimization choice.

**Why is R2 negative for the short baseline?**  The model's test predictions were too concentrated around the central tendency and generalized poorly after only two epochs.

**Does 24 hours prove the best possible context?**  No. It performed best among the four tested sequence lengths.
