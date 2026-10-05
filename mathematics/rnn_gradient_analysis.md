# Vanilla RNN Vanishing and Exploding Gradient Analysis

## 1. RNN recurrence

The validated vanilla RNN uses column vectors and the equations:

```text
a_t = W_xh x_t + W_hh h_(t-1) + b_h
h_t = tanh(a_t)
y_hat = W_hy h_T + b_y
```

The recurrent term `W_hh h_(t-1)` makes the current hidden state depend on
the previous hidden state. The same `W_hh` is reused at every timestep.

## 2. Jacobian convention and recurrent gradient

All hidden states and their gradients are treated as column vectors. A small
change in the previous hidden state gives:

```text
da_(t+1) = W_hh dh_t
```

Let:

```text
D_(t+1) = diag(1 - h_(t+1)^2)
```

because `tanh'(a_(t+1)) = 1 - tanh(a_(t+1))^2`. Therefore:

```text
dh_(t+1) = D_(t+1) W_hh dh_t
```

The Jacobian mapping `h_t` to `h_(t+1)` is thus:

```text
J_(t+1) = D_(t+1) W_hh
```

For column-vector gradients, the chain rule uses the transpose Jacobian:

```text
dL/dh_t = J_(t+1)^T dL/dh_(t+1)
         = W_hh^T D_(t+1) dL/dh_(t+1)
```

This is equivalent to writing the product in the opposite orientation when
using row-vector gradients.

## 3. Repeated matrix multiplication

Moving from timestep `T` back to an earlier timestep repeatedly applies these
Jacobians:

```text
dL/dh_t = J_(t+1)^T J_(t+2)^T ... J_T^T dL/dh_T
```

Equivalently, ignoring orientation and focusing on magnitude:

```text
||dL/dh_t|| is controlled by the product of D_k W_hh terms,
for k = t+1, ..., T.
```

Repeated multiplication can shrink the gradient toward zero or enlarge it
exponentially. This is the mathematical source of the vanishing/exploding
gradient vulnerability.

## 4. Scalar intuition

These simplified values illustrate repeated multiplication; they are not the
exact gradients of this trained model:

```text
0.5^10 = 0.0009765625
0.5^20 = 0.00000095367431640625

1.5^10 = 57.6650390625
1.5^20 = 3325.256730079651
```

An effective derivative below one produces decay, while one above one produces
growth. In the real network, the effective factors are matrix Jacobians and
differ from unit to unit and timestep to timestep.

## 5. Tanh contribution

```text
tanh'(a_t) = 1 - tanh(a_t)^2
```

For finite `a_t`, the derivative satisfies `0 < tanh'(a_t) <= 1`. When a unit
saturates near `tanh(a_t) = 1` or `tanh(a_t) = -1`, the derivative approaches
zero:

```text
1 - tanh(a_t)^2 ~= 0
```

The diagonal matrix `D_t` therefore attenuates the recurrent gradient in
`W_hh^T D_(t+1) dL/dh_(t+1)`. This attenuation can compound across many
timesteps.

## 6. Actual saved-model diagnostics

The analysis used `results/rnn_numpy_model.npz`, the 2-epoch baseline, and the
first real test sequence with shape `(24, 7)`. These diagnostics describe the
recurrent matrix but do not by themselves prove vanishing or exploding
behavior.

- `W_hh` shape: `[32, 32]`
- Frobenius norm: `0.32363060`
- Spectral norm: `0.11037690`
- Minimum singular value: `0.00120621`
- Maximum singular value: `0.11037690`

Singular values:

```text
0.11037690, 0.10240478, 0.09615088, 0.09223791, 0.08634463, 0.08376181, 0.07897238, 0.07544527, 0.07174325, 0.06718189, 0.06475643, 0.05967943, 0.05613015, 0.05267536, 0.04846621, 0.04661883, 0.04351856, 0.03935445, 0.03703756, 0.03551279, 0.03006638, 0.02811222, 0.02455032, 0.02149707, 0.02117717, 0.01645929, 0.01316964, 0.01221799, 0.01044705, 0.00593495, 0.00532483, 0.00120621
```

## 7. Empirical gradient propagation

The starting gradient was `dL/dh_T = W_hy^T (y_hat - y)`. For each timestep,
the analysis recorded the column-vector norms before applying the local tanh
derivative, then propagated with:

```text
dL/dh_(t-1) = W_hh^T [dL/dh_t * (1 - h_t^2)]
```

The recorded values are listed in backward order, `t = 23, ..., 0`:

| t | ||dL/dh_t|| | ||dL/da_t|| |
|---:|---:|---:|
| 23 | 4.97363813e-04 | 4.96969535e-04 |
| 22 | 3.01627767e-05 | 3.01538994e-05 |
| 21 | 1.64760752e-06 | 1.64643163e-06 |
| 20 | 7.99936383e-08 | 7.99674144e-08 |
| 19 | 4.83881432e-09 | 4.83608968e-09 |
| 18 | 2.84350844e-10 | 2.84237187e-10 |
| 17 | 1.97316550e-11 | 1.97142708e-11 |
| 16 | 1.00222702e-12 | 1.00186102e-12 |
| 15 | 4.27642681e-14 | 4.27491582e-14 |
| 14 | 2.27511239e-15 | 2.27461568e-15 |
| 13 | 1.56474848e-16 | 1.56469749e-16 |
| 12 | 1.03539935e-17 | 1.03536173e-17 |
| 11 | 6.07256859e-19 | 6.07229256e-19 |
| 10 | 3.38572844e-20 | 3.38528835e-20 |
| 9 | 1.96426608e-21 | 1.96419675e-21 |
| 8 | 1.12628310e-22 | 1.12624434e-22 |
| 7 | 6.73228370e-24 | 6.73183959e-24 |
| 6 | 3.89757123e-25 | 3.89742894e-25 |
| 5 | 2.55486286e-26 | 2.55476786e-26 |
| 4 | 1.74415562e-27 | 1.74383100e-27 |
| 3 | 1.11945424e-28 | 1.11893771e-28 |
| 2 | 7.11633541e-30 | 7.11585450e-30 |
| 1 | 4.08942221e-31 | 4.08933237e-31 |
| 0 | 2.39827029e-32 | 2.39812231e-32 |


The plot is saved as `results/rnn_gradient_propagation.png` with the vertical
axis `log10(gradient norm)`.

## 8. Interpretation and limitations

This sequence shows strong variation in gradient magnitude across the 24 timesteps; the ratio of largest to smallest hidden-state gradient norm is greater than 1e6. This is evidence of severe gradient scale variation for this checkpoint and sequence, not a universal claim about every sequence.

Vanilla RNNs are susceptible to vanishing/exploding gradients because the
recurrent gradient involves repeated multiplication of Jacobian terms. The
severity depends on the recurrent weights, activation derivatives, sequence
length, and learned state dynamics. This one sequence and one saved checkpoint
cannot establish behavior for every sequence or every training stage.
