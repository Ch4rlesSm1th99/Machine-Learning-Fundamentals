"""
This code demonstrates logistic regression on synthetic data. The main steps include:
1. Generating random data points for two classes.
2. Visualizing the generated data points.
3. Implementing logistic regression:
    a. Forward propagation to compute predicted probabilities.
    b. Computation of loss using the log loss function.
    c. Backward propagation to compute gradients.
    d. Weight update using gradient descent.
4. Visualizing the decision boundary at various iterations.
5. Plotting the loss curve across iterations.
6. Comparing original labels with the predicted labels.

The visualization assists in understanding how the decision boundary changes as the algorithm learns from the data.
"""

import numpy as np
import matplotlib.pyplot as plt

# Function to plot data points
def plotClass(X, y, p):
    '''Plot the data points with class colors.'''
    for i in range(y.shape[1]):
        if y[0, i] == 0:
            plt.plot(X[0, i], X[1, i], 'r' + p)
        else:
            plt.plot(X[0, i], X[1, i], 'b' + p)

# Generate synthetic data
num_data = 100  # data points per class

x1 = np.random.randn(2, num_data) + 2
x0 = np.random.randn(2, num_data)
y1 = np.ones((1, num_data))
y0 = np.zeros((1, num_data))

# Combine the data
X = np.concatenate((x1, x0), axis=1)
y = np.concatenate((y1, y0), axis=1)

# Initial plot
plotClass(X, y, 'o')
plt.show()

# Pre-processing the data for gradient descent
A = X.T
x_ones = np.ones((A.shape[0], 1))
A = np.concatenate([x_ones, A], axis=1)  # Appending ones for the bias term

# Functions for forward propagation
def get_z(X, w):
    '''Calculate linear combination of weights and features.'''
    return X @ w

def sigmoid(z):
    '''Apply the sigmoid activation function.'''
    return 1 / (1 + np.exp(-z))

def loss(yhat, y):
    '''Compute the logistic regression loss.'''
    return np.mean(-y * np.log(yhat) - (1-y) * np.log(1-yhat))

# Gradient descent
w = np.random.randn(A.shape[1], 1)
lr = 2e-7
ls = []
it_list = [0, 10, 50, 100, 1000, 5000]
for i in range(10000):
    # Forward pass
    z = get_z(A, w)
    yhat = sigmoid(z)
    l = loss(yhat, y)
    ls.append(l)

    # Backward pass
    dz = yhat - y.T
    dw = A.T @ dz

    # Update weights
    w = w - lr * dw

    # Plot the decision boundary at specified iterations
    if i in it_list:
        labels = yhat > 0.5
        plt.figure()
        plotClass(X, y, 'o')
        u = np.linspace(-2, 3, 10)
        v = (-w[0, 0] - w[1, 0] * u) / w[2, 0]
        plt.plot(u, v, "g")
        plt.title(f"Iteration {i}")
        plt.show()

# Plot loss curve
plt.figure()
plt.plot(ls)
plt.title('Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Compute predictions
labels = 1.0 * (yhat > 0.5)

# Compare original labels and predictions side by side
plt.figure(figsize=(10, 5))

# Original labels
plt.subplot(121)
plotClass(X, y, '*')
plt.title("Original Labels")

# Predicted labels
plt.subplot(122)
plotClass(X, labels.T, 'o')
plt.title("Predicted Labels")

plt.tight_layout()
plt.show()
