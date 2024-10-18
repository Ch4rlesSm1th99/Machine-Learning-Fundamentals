'''
This script performs binary classification on synthetic data using logistic regression. It first generates 2D points,
classifying them into two groups based on their radial distance from the origin. These points are visualized in a
scatter plot, differentiated by class. The script constructs a logistic regression model, extending the data's feature
space with polynomials to capture complex boundaries. The model is trained via gradient descent, minimizing
cross-entropy loss, with the training's efficacy demonstrated through a plot of loss over iterations. Post-training,
the model's predictions are visualized on the data points. Additionally, the script applies scikit-learn's logistic
regression in a pipeline with polynomial feature expansion, training it on the same data, and visualizing the
predictions for comparison. Essentially, the script is a demonstration of building, training, and visualizing a binary
classification model, alongside a comparison with scikit-learn's implementation.
'''


import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.preprocessing as preprocessing
import sklearn.pipeline as pipeline

# Function to plot data points with different classes
def plotClass(X, y, p, title):
    plt.figure()  # Create a new figure
    plt.title(title)  # Set the title of the plot
    # Loop over each data point
    for i in range(X.shape[1]):
        color = 'r' if y[0, i] == 0 else 'b'  # Choose color based on class
        plt.plot(X[0, i], X[1, i], color + p)  # Plot the data point
    plt.show()  # Display the plot

num_data = 1000  # Number of data points

# Generate random data
X = np.random.uniform(-1, 1, [2, num_data])  # Randomly generated features
# Generate labels based on a condition, resulting in a circular decision boundary
y = (X[0, :]**2 + X[1, :]**2 - 0.5 > 0).astype(np.int8)[None, :]

# Plot original data
plotClass(X, y, 'o', "Original Data")

# Function to calculate z (input for sigmoid function)
def get_z(W, X):
    return np.dot(W, X)  # Matrix multiplication

# Sigmoid activation function
def get_sigmoid(z):
    return 1 / (1 + np.exp(-z))  # Sigmoid formula

# Loss function (Cross-Entropy)
def get_loss(y, yhat):
    m = y.shape[1]  # Number of samples
    # Cross-entropy loss formula
    return -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)) / m

nb = 3  # Degree for polynomial features
A = []  # List to hold polynomial features
# Generate polynomial features based on the degree
for i in range(nb):
    for j in range(nb):
        A.append(X[0, :]**i * X[1, :]**j)

A = np.array(A)  # Convert list to numpy array

ls = []  # List to keep track of loss values
lr = 0.01  # Learning rate for gradient descent
W = np.random.randn(1, A.shape[0])  # Initialize weights randomly

# Training loop
for i in range(10000):  # Number of iterations
    z = get_z(W, A)  # Get z using current weights
    yhat = get_sigmoid(z)  # Get predictions using sigmoid activation
    loss = get_loss(y, yhat)  # Calculate loss

    ls.append(loss)  # Append current loss to the list

    dZ = yhat - y  # Calculate derivative of loss with respect to z
    dW = (1. / num_data) * np.dot(dZ, A.T)  # Calculate derivative of loss with respect to weights

    W -= lr * dW  # Update weights using gradient descent

# Plotting loss over iterations
plt.figure()
plt.title("Loss over iterations")
plt.plot(ls)

# Plot data points classified by the trained model
plotClass(X, yhat > 0.5, 'x', "Trained Model Predictions")

# Using sklearn's logistic regression
pipeline = pipeline.Pipeline([
    ('basis', preprocessing.PolynomialFeatures(3)),  # Polynomial features of degree 3
    ('model', lm.LogisticRegression(max_iter=10000))  # Logistic regression model
])
pipeline.fit(X.T, y.T.ravel())  # Fit the model to the data

yhat_sklearn = pipeline.predict(X.T)  # Make predictions

# Plot data points classified by sklearn's logistic regression
plotClass(X, yhat_sklearn[None, :], 'o', "Sklearn Model Predictions")