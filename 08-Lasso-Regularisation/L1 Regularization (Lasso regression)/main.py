'''
L1 regularization, also known as Lasso, is a technique that helps prevent overfitting and enhances a model's
generalization by adding a penalty to the loss function. This penalty is proportional to the absolute value of the
model coefficients. The effect is that it can shrink some coefficients to zero, effectively performing feature
selection by eliminating less important features. It simplifies the model, making it more interpretable and focused
on the most relevant features.
'''


import numpy as np  # Importing the numpy library for numerical operations
import matplotlib.pyplot as plt  # Importing the matplotlib library for plotting and visualization


# Function to generate artificial data
def generate_data(n_samples=100):
    np.random.seed(0)  # Setting seed for reproducibility
    X = 2 - 3 * np.random.normal(0, 1, n_samples)  # Generating X with random data
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-3, 3,
                                                             n_samples)  # Generating y with a polynomial relationship to X plus noise
    X = X[:, np.newaxis]  # Reshaping X to a 2D array for compatibility
    y = y[:, np.newaxis]  # Reshaping y to a 2D array for compatibility
    return X, y


# Function to normalize features
def normalize_features(X):
    mean = np.mean(X, axis=0)  # Calculating mean of X
    std_dev = np.std(X, axis=0)  # Calculating standard deviation of X
    X_normalized = (X - mean) / std_dev  # Normalizing X
    return X_normalized


# Function to compute the L1 norm
def l1_norm(weights):
    return np.sum(np.abs(weights))  # Computing L1 norm of weights


# Function for polynomial regression
def polynomial_regression(X, y, degree, l1_penalty, lr, n_iterations):
    # Extend X with polynomial terms
    X_poly = X  # Initializing the polynomial feature matrix with X
    for d in range(2, degree + 1):  # Iterating over degrees from 2 to the specified degree
        X_poly = np.hstack((X_poly, np.power(X, d)))  # Adding polynomial features to X_poly

    # Normalize features
    X_poly = normalize_features(X_poly)  # Normalizing polynomial features

    # Add intercept term to X
    X_poly = np.hstack(
        (np.ones((X_poly.shape[0], 1)), X_poly))  # Adding a column of ones to X_poly to account for the intercept

    # Initialize weights
    weights = np.random.randn(X_poly.shape[1], 1)  # Initializing weights randomly

    # Gradient descent
    for i in range(n_iterations):  # Iterating over the number of specified iterations
        predictions = np.dot(X_poly, weights)  # Calculating predictions
        residuals = predictions - y  # Calculating residuals
        gradients = 2 / X.shape[0] * np.dot(X_poly.T, residuals) + l1_penalty * np.sign(
            weights)  # Calculating gradients with the L1 regularization term
        weights -= lr * gradients  # Updating weights using the learning rate and gradients

    return weights, X_poly  # Returning the final weights and the polynomial feature matrix


# Function to plot predictions
def plot_predictions(X, y, weights, X_poly):
    plt.scatter(X, y, color='blue', s=30, marker='o', label="Input data")  # Plotting the original data points

    # Create continuous line to plot
    X_continuous = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)  # Creating a range of values from min to max of X
    X_continuous_poly = X_continuous  # Initializing the polynomial feature matrix for X_continuous
    for d in range(2, X_poly.shape[1]):  # Iterating over the degree of polynomials
        X_continuous_poly = np.hstack(
            (X_continuous_poly, np.power(X_continuous, d)))  # Adding polynomial features to X_continuous_poly
    X_continuous_poly = normalize_features(X_continuous_poly)  # Normalizing features of X_continuous_poly
    X_continuous_poly = np.hstack(
        (np.ones((X_continuous_poly.shape[0], 1)), X_continuous_poly))  # Adding intercept term to X_continuous_poly

    predictions = np.dot(X_continuous_poly, weights)  # Making predictions using the final weights
    plt.plot(X_continuous, predictions, color='red',
             label="Fitted polynomial regression")  # Plotting the regression line
    plt.xlabel("X")  # Adding label for the x-axis
    plt.ylabel("y")  # Adding label for the y-axis
    plt.title("Polynomial Regression with L1 regularization")  # Adding title for the plot
    plt.legend(loc='upper right')  # Adding legend to the plot
    plt.show()  # Displaying the plot


# Generate artificial data
X, y = generate_data(100)  # Generating artificial data

# Set hyperparameters
degree = 2  # Degree of the polynomial regression
l1_penalty = 0.1  # L1 regularization penalty
lr = 0.01  # Learning rate for gradient descent
n_iterations = 10000  # Number of iterations for gradient descent

# Perform polynomial regression
weights, X_poly = polynomial_regression(X, y, degree, l1_penalty, lr, n_iterations)  # Performing polynomial regression

# Plot predictions
plot_predictions(X, y, weights, X_poly)  # Plotting predictions against the original data

# Print L1 norm of the weights
print(f"L1 norm of weights: {l1_norm(weights)}")  # Printing the L1 norm of the final weights

# Print final weights
print("Final weights are:", weights)  # Printing the final weights
