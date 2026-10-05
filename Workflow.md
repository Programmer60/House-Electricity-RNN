You are my AI coding/research assistant for a college project titled:

**"Household Electricity Consumption Forecasting Using Recurrent Neural Networks"**

## 1. Project Context

We are a 3-member team with approximately 11 days to complete the project.

The project uses the UCI Individual Household Electric Power Consumption dataset.

The overall project pipeline is:

Raw Dataset
→ Preprocessing in R
→ Hourly Time Series
→ Normalization
→ Train/Test Split
→ Sliding Window Sequences
→ Vanilla RNN
→ LSTM extension
→ Evaluation
→ Mathematical Analysis
→ RNN vs LSTM Comparison

I am **Member 1**.

My responsibility is specifically:

* Vanilla RNN implementation
* RNN mathematical analysis
* Forward propagation
* Loss calculation
* Backpropagation Through Time (BPTT)
* Gradient descent
* Vanishing/exploding gradient analysis
* RNN experiments
* RNN results and visualizations
* RNN section of the report/PPT

Another team member is responsible for LSTM.
Another team member is responsible for preprocessing/evaluation/integration.

Therefore, DO NOT duplicate the entire preprocessing pipeline in Python.

---

# 2. Important Technology Decision

The preprocessing MUST remain in R.

The R preprocessing pipeline produces:

* train_sequences.rds
* test_sequences.rds
* preprocessing_metadata.rds

Python is used only for the RNN/modeling side.

The expected processed data format is:

X_train: (N, 24, 7)
y_train: (N, 1)

X_test: (N, 24, 7)
y_test: (N, 1)

Meaning:

* 24 = previous 24 hourly observations
* 7 = numerical electricity features
* target = next-hour Global_active_power

The sequence relationship must be:

X[i] = observations from t-23 through t
y[i] = Global_active_power at t+1

DO NOT assume this is correct without verifying it from the actual R preprocessing output.

---

# 3. First Task — Inspect Before Coding

Before writing the RNN model:

1. Inspect the available project files.
2. Find the R preprocessing outputs.
3. Determine the exact structure of:

   * train_sequences.rds
   * test_sequences.rds
   * preprocessing_metadata.rds
4. Verify:

   * dimensions
   * feature ordering
   * target variable
   * scaling method
   * train/test split
   * sequence alignment
5. Specifically check for an off-by-one error between the input window and prediction target.

Do not silently assume the data format.

If something is inconsistent, explain the issue before modifying anything.

---

# 4. Python Environment

Use Python for the RNN.

Initially implement the core RNN using **NumPy** rather than immediately hiding the mathematics inside a deep-learning framework.

The purpose is to connect the mathematical equations directly to the implementation.

After the NumPy implementation is verified, create a practical training implementation using PyTorch if useful.

Use:

* Python
* NumPy
* Pandas where useful
* Matplotlib
* scikit-learn only for evaluation utilities if needed
* PyTorch for the practical neural-network implementation

Do NOT use TensorFlow/Keras unless there is a specific reason.

---

# 5. RNN Architecture

Start with a simple vanilla RNN:

Input
→ Simple RNN
→ Dense output

Baseline:

Input shape:
(24, 7)

Hidden units:
32

Output:
1

Activation:
tanh for hidden state

Output activation:
linear

Loss:
Mean Squared Error

Initial learning rate:
0.001 for the practical implementation

The 32 hidden units are a baseline hyperparameter, NOT a theoretically optimal value.

---

# 6. Mathematical Foundation

The mathematical analysis is a major requirement of this project.

Explain and implement the following equation:

For each timestep:

h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)

Output:

y_hat = W_hy h_T + b_y

Loss:

L = (1/N) Σ(y_i - y_hat_i)^2

Explain:

* dimensions of every variable
* meaning of every weight matrix
* hidden state
* recurrent connection
* activation function
* output layer
* loss function

For the baseline with 7 input features and 32 hidden units, explicitly verify the dimensions:

x_t ∈ R^7

h_t ∈ R^32

W_xh ∈ R^(32×7)

W_hh ∈ R^(32×32)

b_h ∈ R^32

W_hy ∈ R^(1×32)

b_y ∈ R

Calculate the total number of trainable parameters and verify it programmatically.

---

# 7. Forward Propagation

Implement forward propagation explicitly.

For each of the 24 timesteps:

1. Take x_t
2. Compute:

a_t = W_xh x_t + W_hh h_(t-1) + b_h

3. Compute:

h_t = tanh(a_t)

4. Continue to the next timestep.

After the final timestep:

y_hat = W_hy h_T + b_y

The implementation must make the relationship between the equations and code obvious.

Include comments explaining the mathematical operation rather than unnecessary comments describing obvious Python syntax.

---

# 8. Backpropagation Through Time

This is extremely important.

Do not simply call it "backpropagation."

Explain **Backpropagation Through Time (BPTT)**.

Show how the recurrent network is conceptually unrolled:

x_1 → h_1 → h_2 → ... → h_24 → y_hat

Explain why the same weights are reused at every timestep.

Derive the main gradient relationships using the chain rule.

In particular, explain how gradients propagate through:

* output layer
* hidden state
* recurrent connection
* tanh derivative

Use:

d/dx tanh(x) = 1 - tanh²(x)

Explain why gradients accumulate across multiple timesteps.

Do not skip directly to an optimizer without understanding the gradient calculation.

---

# 9. Vanishing and Exploding Gradients

Provide a mathematical explanation.

Show how repeated multiplication of derivatives/Jacobian terms through time can cause:

Vanishing gradients:

|gradient| → 0

Exploding gradients:

|gradient| → very large

Use a simple numerical example such as:

0.5^10

and

2^10

to demonstrate the intuition.

Explain why this is an important limitation of vanilla RNNs.

This explanation will later motivate the LSTM extension.

---

# 10. Training

First create a small sanity-check training example to verify that the NumPy RNN implementation can learn.

Then train the actual household electricity dataset.

Track:

* training loss
* validation loss if a validation split is available/appropriate
* epoch number

Plot the loss curve.

Do not introduce unnecessary complexity such as bidirectional RNNs, attention, transformers, etc.

The project specifically needs a vanilla RNN foundation.

---

# 11. Experiments

After the baseline works, conduct controlled experiments.

Experiment A — Hidden units:

16
32
64

Experiment B — Sequence length:

6
12
24
48

Only perform experiments that are computationally reasonable.

Keep the dataset, preprocessing, evaluation methodology and other hyperparameters consistent when comparing models.

Record results systematically.

---

# 12. Evaluation

Evaluate predictions using:

MAE
MSE
RMSE
R²

Because the target was normalized during preprocessing, make sure predictions and actual values are converted back to the original scale before reporting metrics in kW.

Do not report a normalized RMSE as if it were kW.

Create:

1. Actual vs predicted plot
2. Training loss curve
3. Experiment comparison table

Clearly label units.

Remember:

Global_active_power is measured in **kW**, not kWh.

---

# 13. Reproducibility

Set random seeds.

Record:

* random seed
* hidden units
* sequence length
* learning rate
* number of epochs
* batch size
* optimizer
* loss function

Save the final model and experiment results.

Use a clean project structure such as:

project/
│
├── data/
├── notebooks/
├── models/
│   ├── rnn_numpy.py
│   └── rnn_pytorch.py
│
├── experiments/
├── results/
├── mathematics/
│   └── rnn_math.md
└── README.md

---

# 14. Coding Standards

Write clean, modular Python.

Separate:

* data loading
* data validation
* model definition
* training
* prediction
* evaluation
* visualization

Do not create one giant Python script.

Use functions/classes where appropriate.

Avoid unnecessary libraries.

Do not hide important mathematical operations behind high-level APIs in the NumPy implementation.

---

# 15. Academic Integrity / Explanation Requirement

I need to be able to explain every important part of the code to my teacher.

Therefore, whenever you introduce an important piece of code, explain:

1. What mathematical equation it represents.
2. What the input dimensions are.
3. What the output dimensions are.
4. Why it is required.
5. How it relates to the RNN theory.

Do not give me code that I cannot explain.

If you make a design choice, state whether it is:

* mathematically required,
* a standard practice,
* or simply a chosen hyperparameter.

For example:

"32 hidden units" is a hyperparameter, not a mathematical requirement.

---

# 16. Development Strategy

Work incrementally.

Do NOT generate the entire project blindly in one step.

Follow this order:

STEP 1
Inspect and validate the R `.rds` outputs.

STEP 2
Load the data into Python.

STEP 3
Verify dimensions and target alignment.

STEP 4
Implement a tiny NumPy RNN on synthetic data.

STEP 5
Verify forward propagation manually.

STEP 6
Implement the loss.

STEP 7
Implement BPTT and gradient calculation.

STEP 8
Verify gradients using a numerical gradient check.

STEP 9
Train the NumPy RNN.

STEP 10
Implement the practical PyTorch RNN.

STEP 11
Train on the actual electricity dataset.

STEP 12
Run controlled experiments.

STEP 13
Generate evaluation metrics and plots.

STEP 14
Prepare the mathematical analysis.

STEP 15
Prepare the RNN report/PPT section.

At every stage, test the current component before moving to the next stage.

---

# 17. Important Constraints

Do NOT:

* redo preprocessing in Python
* randomly shuffle the time-series data
* leak test information into training
* fit normalization parameters using the test set
* use future observations to predict the past
* claim 32 hidden units are optimal without experiments
* claim LSTM is better without experimental evidence
* report kWh when the target is Global_active_power in kW
* skip checking the sequence target alignment
* blindly copy code without explaining the mathematics

---

# 18. Final Goal

The final project should allow me to explain this complete chain to my teacher:

Raw electricity measurements
↓
R preprocessing
↓
Hourly time series
↓
24-hour input sequence
↓
x_t
↓
RNN hidden-state equation
↓
h_t
↓
Output equation
↓
Prediction
↓
MSE
↓
BPTT
↓
Gradients
↓
Weight update
↓
Learning
↓
Electricity forecast
↓
Evaluation

The RNN should be the main mathematical model.

LSTM will be handled as a separate extension by another team member, so do not implement LSTM unless explicitly asked.

Start with **Step 1: inspect the available R preprocessing outputs and verify their exact structure. Do not start model training until the data interface is verified.**
