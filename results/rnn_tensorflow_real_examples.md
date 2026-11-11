# Trained RNN: Architecture and Real Test Examples

These examples use the saved TensorFlow/Keras checkpoint and three actual held-out test sequences. Cases are selected near the 15th, 50th, and 98th percentiles of actual next-hour demand; they are not synthetic inputs.

## Model architecture

```text
Input: 24 previous hours x 7 training-scaled features
  -> SimpleRNN(32, tanh): 32 hidden activations at each hour
  -> use final hidden state h_24
  -> Dense(1): next-hour prediction in training-scaled units
  -> inverse min-max transform: prediction in kW
```

The recurrent layer reuses the same input weights, recurrent weights, and bias at all 24 hours. The RNN layer has 1,280 parameters; the dense output layer has 33, for 1,313 total.

## Three real held-out predictions

| Case | Test sample | Actual (kW) | Predicted (kW) | Error: actual - predicted (kW) |
|---|---:|---:|---:|---:|
| Low demand (15th percentile) | 526 | 0.3309 | 0.6159 | -0.2850 |
| Typical demand (50th percentile) | 1 | 0.7734 | 1.0022 | -0.2288 |
| High demand (98th percentile) | 5212 | 2.8731 | 2.0361 | +0.8371 |

## How to explain the visualizations

- **Architecture diagram:** each input hour contributes seven measurements. The same SimpleRNN cell updates the hidden state 24 times; the final state is passed to the Dense output layer.
- **Feature-window heatmap:** shows the seven training-scaled feature values the model actually received for each selected sequence.
- **Prediction and hidden-state panels:** the left plots show the 24-hour Global_active_power history and compare actual versus predicted next-hour demand. The right heatmaps show how all 32 hidden units changed during the sequence.
- **Important distinction:** hidden activations are learned internal representations, not direct appliance labels or causal explanations.

## Run again

```text
python models/rnn/visualize_tensorflow_examples.py
```

The script writes the architecture figure, the two real-example figures, this report, and a CSV of the three measured predictions under `results/`.
