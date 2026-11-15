# BPTT and Gradient Analysis

## Unrolling

An RNN is conceptually unrolled through time:

```text
x_0 -> h_0 -> h_1 -> ... -> h_T -> y_hat
```

The same weights are reused at every timestep. BPTT applies the chain rule backward through every reused recurrent connection.

## Jacobian

Using column-vector gradients:

```text
D_(t+1) = diag(1 - h_(t+1)^2)
J_(t+1) = D_(t+1) W_hh
```

Therefore:

```text
dL/dh_t = W_hh^T D_(t+1) dL/dh_(t+1)
```

Repeated products of these Jacobians control how much gradient reaches early timesteps.

## Vanishing and exploding gradients

A scalar effective derivative of `0.5` gives:

```text
0.5^10 = 0.0009765625
0.5^20 = 0.00000095367431640625
```

An effective derivative of `1.5` gives:

```text
1.5^10 = 57.6650390625
1.5^20 = 3325.256730079651
```

The tanh derivative is:

```text
tanh'(a_t) = 1 - tanh(a_t)^2
```

When tanh saturates near `-1` or `1`, this derivative approaches zero and can attenuate gradients.

## Measured result

For the saved NumPy baseline and one real 24-timestep sequence, the hidden gradient norm decreased from approximately `4.97e-4` to `2.40e-32` while propagating backward. This demonstrates severe vanishing for that checkpoint and sequence, not a universal claim about every RNN.

See [the full gradient analysis](../mathematics/rnn_gradient_analysis.md) and [the propagation plot](../results/rnn_gradient_propagation.png).
