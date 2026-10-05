# NumPy and PyTorch Vanilla RNN Comparison

| Model | Epochs | Optimizer | Test MAE (kW) | Test RMSE (kW) | Test R^2 |
|---|---:|---|---:|---:|---:|
| NumPy RNN, 2 epochs | 2 | SGD | 0.69906987 | 0.81046615 | -0.23954534 |
| NumPy RNN, 20 epochs | 20 | SGD | 0.48833551 | 0.59847853 | 0.32408855 |
| PyTorch RNN | 11 | Adam | 0.39160449 | 0.51785915 | 0.49392373 |
