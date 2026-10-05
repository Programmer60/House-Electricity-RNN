# Baseline NumPy Vanilla RNN Evaluation

## Baseline configuration

- Model: NumPy Vanilla RNN
- Input: `24 x 7`
- Hidden state: `h_t in R^32`
- Output: `y_hat in R`
- Optimizer: plain SGD
- Learning rate: `0.001`
- Epochs: `2`
- Random seed: `42`

The model was trained only on the chronological training set. The test set was
kept unseen: evaluation performs forward propagation only and does not update
any parameters. Predictions and targets were inverse-transformed with the
training-only `Global_active_power` min/max values from
`preprocessing_metadata.rds`. All reported metrics use the original kW unit.

## Test metrics

- MAE: `0.69906987` kW
- MSE: `0.65685538` kW^2
- RMSE: `0.81046615` kW
- R^2: `-0.23954534`

Prediction range: `1.35474566` to `1.56703428` kW  
Actual target range: `0.16043757` to `5.62683305` kW

These results describe one specific baseline configuration and are not an
optimized-model claim. Training behavior, unseen-test metrics, the
actual-versus-predicted plot, and the signed-error plot should be considered
together.
