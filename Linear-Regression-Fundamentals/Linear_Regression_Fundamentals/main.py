"""
Linear Regression 1 (1 feature : 1 label and linear fit)

When using linear regression to model data we are given some data and we solve the normal
equation to find a linear combination of our weights which minimises our loss function
and then use those weights to make predictions. (The normal equation can be derived by
taking mean squared error and setting its derivative to zero then some linear algebra).

General steps to implement a Linear Regression model are:

1. splitting data (train set, test set (validation set as well potentially))

2. using training features solve the normal equation to find your weights

3. using your weights (normal eq solution) make predictions for labels

4. find loss of model on the test set

5. further analysis / tweaking (ie.feature engineering?, regularisation?, change of basis type?)
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(64)

'''generate our data that we are going to try modeling'''
# number of samples
num_samples = 500

# generating x data
x_data = np.linspace(0, 10, num_samples)

# defining our trend parameters
slope = 2.5
intercept = 1.5
noise_strength = 10
noise = np.random.rand(num_samples)

# generating y data
y_data = slope * x_data + intercept + noise_strength * noise

'''
split the data set into training set and testing set here I'm going to use numpy but 
a much more efficient way is using sk learns split function which will be used in other examples.
'''


# splits data 80:20 train:test sets
def split_data(data_x, data_y):
    # randomise indexes
    ind = np.arange(data_x.shape[0])  # creates an array of indexes(in order)
    np.random.shuffle(ind)  # shuffles the indexes

    # split the indexes
    train_ind = ind[:int(np.floor(0.8 * data_x.shape[0]))]  # assign indexes to training data
    test_ind = ind[int(np.floor(0.8 * data_x.shape[0])):]  # assign indexes to testing data

    # sample those indexes from each set of data
    x_train = data_x[train_ind]
    y_train = data_y[train_ind]

    x_test = data_x[test_ind]
    y_test = data_y[test_ind]

    return x_train, y_train, x_test, y_test


# assign our datasets
x_training_data, y_training_data, x_testing_data, y_testing_data = split_data(x_data, y_data)

plt.plot(x_training_data, y_training_data, "bx", label="training data")
plt.plot(x_testing_data, y_testing_data, "rx", label="testing data")
plt.legend()
plt.title("Training Data vs Test Data")
plt.grid()
plt.show()

'''
solve the normal equation for our weights to do this we need to introduce a bias term to our x_data to account for the
intercept. Since the intercept for every point is a constant we add a column of ones on the left hand side of our x data 
because the bias term for each point will just be the weight on its own.

Normal Equation:
        theta = (X.T @ X)^-1 . (X.T @ y)
'''

# adds our bias term to our x training set
x_training_matrix = np.concatenate((np.ones((x_training_data.shape[0], 1)), x_training_data.reshape(-1, 1)), axis=1)

# solve normal equation
theta = np.linalg.solve(x_training_matrix.T @ x_training_matrix, x_training_matrix.T @ y_training_data)

# compute x matrix for testing data
x_testing_matrix = np.concatenate((np.ones((x_testing_data.shape[0], 1)), x_testing_data.reshape(-1, 1)), axis=1)

# make predictions using testing data
y_hat = x_testing_matrix @ theta

plt.plot(x_testing_data, y_testing_data, "bx", label="testing data")
plt.plot(x_testing_data, y_hat, "r", label="predictions")
plt.legend()
plt.title("Test Data vs Model Predictions")
plt.grid()
plt.show()

# evaluate some kind of loss or error stat
# the mean absolute error
mae = np.mean(np.abs(y_testing_data - y_hat))

print(f"Mean Absolute Error: {mae:.2f}")


