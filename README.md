# ann_assignment
assignmentno.1
I’ve combined the project theory, algorithm, formulas, Python implementation, result, conclusion, and viva questions into one Python file. Save it as `forward_propagation.py` in VS Code.

python
````

"""
============================================================
PROJECT NO: 8043
TITLE: FORWARD PROPAGATION IN A NEURAL NETWORK
============================================================

1. AIM
To implement a feed-forward neural network with 2 input
neurons, 3 hidden neurons, and 1 output neuron using Python
and NumPy. Random weights and biases are initialized, sigmoid
activation is applied, and forward propagation is performed
for three input samples.

2. THEORY
A neural network is a machine learning model made up of
interconnected neurons arranged in layers.

Forward propagation passes input data through the network
to calculate the final output.

Network architecture:
    Input Layer  : 2 neurons
    Hidden Layer : 3 neurons
    Output Layer : 1 neuron

Weights determine the strength of connections between
neurons. Biases are additional values added to weighted sums.

Sigmoid activation function:
    sigmoid(z) = 1 / (1 + exp(-z))

Hidden layer:
    Z1 = X * W1 + b1
    H  = sigmoid(Z1)

Output layer:
    Z2 = H * W2 + b2
    Y  = sigmoid(Z2)

Here:
    X  = Input data
    W1 = Input-to-hidden weights
    b1 = Hidden-layer biases
    H  = Hidden-layer activations
    W2 = Hidden-to-output weights
    b2 = Output-layer bias
    Y  = Final output

3. ALGORITHM
    1. Import NumPy.
    2. Define the network architecture.
    3. Initialize weights and biases randomly.
    4. Define the sigmoid activation function.
    5. Create three input samples.
    6. Calculate hidden-layer weighted sums.
    7. Calculate hidden-layer activations.
    8. Calculate the output-layer weighted sum.
    9. Calculate the final output.
    10. Display intermediate values and final outputs.

4. SOFTWARE REQUIREMENTS
    Language     : Python
    Editor       : Visual Studio Code
    Library      : NumPy
    Python       : Python 3.13

5. NOTE
This project demonstrates forward propagation only.
It does not train the network or update the weights.
Randomly initialized weights do not guarantee accurate
predictions.

============================================================
PYTHON IMPLEMENTATION
============================================================
"""

import numpy as np

# ------------------------------------------------------------
# STEP 1: REPRODUCIBLE RANDOM INITIALIZATION
# ------------------------------------------------------------

np.random.seed(42)

# ------------------------------------------------------------
# STEP 2: DEFINE NETWORK ARCHITECTURE
# ------------------------------------------------------------

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# ------------------------------------------------------------
# STEP 3: INITIALIZE WEIGHTS AND BIASES
# ------------------------------------------------------------

# Shape: (2, 3)
W1 = np.random.randn(input_neurons, hidden_neurons)

# Shape: (1, 3)
b1 = np.random.randn(1, hidden_neurons)

# Shape: (3, 1)
W2 = np.random.randn(hidden_neurons, output_neurons)

# Shape: (1, 1)
b2 = np.random.randn(1, output_neurons)

# ------------------------------------------------------------
# STEP 4: DEFINE SIGMOID ACTIVATION FUNCTION
# ------------------------------------------------------------

def sigmoid(z):
    """Convert values into the range 0 to 1."""
    return 1.0 / (1.0 + np.exp(-z))

# ------------------------------------------------------------
# STEP 5: DEFINE FORWARD PROPAGATION
# ------------------------------------------------------------

def forward_propagation(X):

    # Hidden layer weighted sums
    z_hidden = np.dot(X, W1) + b1

    # Hidden layer activations
    hidden_activation = sigmoid(z_hidden)

    # Output layer weighted sum
    z_output = np.dot(hidden_activation, W2) + b2

    # Final output activation
    output = sigmoid(z_output)

    return z_hidden, hidden_activation, z_output, output

# ------------------------------------------------------------
# STEP 6: CREATE THREE INPUT SAMPLES
# ------------------------------------------------------------

X = np.array([
    [0.2, 0.4],
    [0.5, 0.7],
    [0.8, 0.3]
])

# ------------------------------------------------------------
# STEP 7: DISPLAY NETWORK PARAMETERS
# ------------------------------------------------------------

print("=" * 60)
print("PROJECT 8043: FORWARD PROPAGATION")
print("=" * 60)

print("\nNetwork Architecture:")
print("Input neurons  :", input_neurons)
print("Hidden neurons :", hidden_neurons)
print("Output neurons :", output_neurons)

print("\nInput-to-hidden weights (W1):")
print(np.round(W1, 4))

print("\nHidden-layer biases (b1):")
print(np.round(b1, 4))

print("\nHidden-to-output weights (W2):")
print(np.round(W2, 4))

print("\nOutput-layer bias (b2):")
print(np.round(b2, 4))

total_weights = W1.size + W2.size
total_biases = b1.size + b2.size

print("\nTotal weights:", total_weights)
print("Total biases:", total_biases)
print("Total parameters:", total_weights + total_biases)

# ------------------------------------------------------------
# STEP 8: PERFORM FORWARD PROPAGATION FOR EACH SAMPLE
# ------------------------------------------------------------

for i, sample in enumerate(X, start=1):

    # Convert one sample into a two-dimensional array
    sample = sample.reshape(1, input_neurons)

    # Perform forward propagation
    z_h, h, z_o, y_hat = forward_propagation(sample)

    print("\n" + "=" * 60)
    print(f"INPUT SAMPLE {i}")
    print("=" * 60)

    print("\nInput values:")
    print(np.round(sample, 4))

    print("\nHidden-layer weighted sums (Z1):")
    print(np.round(z_h, 4))

    print("\nHidden-layer activations (H):")
    print(np.round(h, 4))

    print("\nOutput-layer weighted sum (Z2):")
    print(np.round(z_o, 4))

    print("\nFinal output (Y):")
    print(np.round(y_hat, 4))

# ------------------------------------------------------------
# STEP 9: DISPLAY RESULT
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print("Forward propagation completed successfully.")
print("All three input samples were processed.")
print("Intermediate activations and final outputs were displayed.")

# ------------------------------------------------------------
# 6. OBSERVATION
#
# Each input sample produces three hidden-layer activations
# and one final output. The output lies between 0 and 1
# because sigmoid activation is used.
#
# 7. ADVANTAGES
#    - Simple and easy to understand.
#    - Demonstrates neural network calculations.
#    - Shows the effects of weights and biases.
#    - Introduces nonlinear activation.
#
# 8. LIMITATIONS
#    - Random weights may produce meaningless predictions.
#    - The network is not trained.
#    - No loss function or weight updates are performed.
#
# 9. CONCLUSION
# A feed-forward neural network with 2 input neurons,
# 3 hidden neurons, and 1 output neuron was implemented
# using Python and NumPy. Random weights and biases were
# initialized, sigmoid activation was applied, and forward
# propagation was demonstrated for three input samples.
#
# 10. VIVA QUESTIONS AND ANSWERS
#
# Q1. What is a neural network?
# A: A computational model made up of interconnected neurons.
#
# Q2. What is forward propagation?
# A: Passing inputs through network layers to calculate output.
#
# Q3. What is a weight?
# A: A parameter that determines connection strength.
#
# Q4. What is a bias?
# A: An additional parameter added to a weighted sum.
#
# Q5. What is sigmoid?
# A: An activation function that produces values between 0 and 1.
#
# Q6. How many weights are used?
# A: Nine weights.
#
# Q7. How many total trainable parameters are used?
# A: Thirteen: nine weights and four biases.
#
# Q8. Does this program train the network?
# A: No. It performs forward propagation only.
#
# Q9. Why is NumPy used?
# A: For arrays, matrix multiplication, and numerical operations.
#
# Q10. What is backpropagation?
# A: A method for calculating gradients to update network
#     parameters during training.
# ------------------------------------------------------------
"""

**How to use it in VS Code**

1. Create a new file and paste the complete code into it.
2. Save it as `forward_propagation.py`.
3. Make sure NumPy is installed. If needed, run `python -m pip install numpy` in the terminal.
4. Click **Run Python File** or execute:

```powershell
python forward_propagation.py
```

The terminal will display the network parameters, intermediate activations, and final outputs for all three samples.
````