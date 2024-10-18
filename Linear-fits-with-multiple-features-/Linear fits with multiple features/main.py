"""
Linear Regression - Multiple Features with a Linear Fit

Linear Regression can be used to fit models on datasets where there are multiple inputs(features) corresponding to a
single output(labels). In this script a linear model is fitted to a dataset comprised of two features 'x1' and 'x2' and
our labels 'y'.

The process is the same with the only difference being that the X matrix in the normal equation will look slightly
different as we need to account for the bias terms (a column of ones), 'x1' (a column of our first features) and 'x2'
(a column of our second features).

The reason I chose two features for this example is so that we could have more than one feature but still be able to
visualise what our data and models look like on a 3d plot.

Linear Regression can still be used data on sets with a much higher number of features than two where we apply the same
logic used in this example where we add a column of that feature onto the end of our X matrix when solving the normal
equation.

Note that linear regression with a very high number of features can only work if we are fitting a very basic function
like a linear fit. If we start to try and fit more complicated functions ie non-linear we will run into 'the curse of
dimensionality' and it will become very computationally expensive and as a result not viable.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split

num_samples = 100

# generate data
x1 = np.linspace(-1, 1, num_samples)
x2 = np.linspace(-1, 1, num_samples)
[X1, X2] = np.meshgrid(x1, x2)  # turns two 1d coord systems into a single 2d coord system

# defining traits of the data
slope1 = 2.5
slope2 = -1.5
intercept = 2.0
noise = np.random.randn(X1.shape[0], X1.shape[1])  # creates noise for rows and columns of data set
strength = 0.5

Y = slope1 * X1 + slope2 * X2 + intercept + noise * strength

# 3d plot of our data
fig = plt.figure(figsize=(12,7))  # initialises plot
ax = fig.add_subplot(111, projection='3d')  # splits figure into 1x1 subplots and adds first subplot to figure = '111'
plot = ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# labeling plot
ax.set_title('original data')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10)  # key

plt.show()

'''
Split the data into train sets and test sets. To do this we are going to combine x1 and x2 together so we have our
features and labels collected together before we split into training and testing data. 
'''

# .ravel flattens data and then .column_stack collects the flattened rows as columns of x1 and x2
X = np.column_stack((X1.ravel(), X2.ravel()))

# splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y.ravel(), test_size=0.2, random_state=42)

# add a bias term to the training and test data
X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# solve normal eq
theta = np.linalg.solve(X_train_bias.T @ X_train_bias, X_train_bias.T @ Y_train)

# predictions on the test data
Y_predictions = X_test_bias @ theta

# mean squared error on test set
mse = ((Y_test - Y_predictions) ** 2).mean()
print("Mean Squared Error on Test Set:", mse)

'''
Note: Next part would not be doable on real dataset but helps to visualise what we are outputting form our data.

In order to visualise the model predictions we going to have to generate predictions over every possible combination of 
x1 and x2 on the meshgrid not just the test data this means that we are going to have to make our predictions by 
substituting in our weights into the initial equation we used to generate the data. Obviously on a real data set 
you wouldn't know the equation/relationship between your features and labels and so this method for visualisation 
wouldn't be viable.
'''

# making predictions for every possible point on the meshgrid this is still the same as above but theta (our weights)
# has been broken into column vectors this time instead.
Y_pred_all_data = theta[0] + theta[1] * X1 + theta[2] * X2

# 3d plot of the original data
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# plotting the surface represented by the model
plot = ax.plot_surface(X1, X2, Y_pred_all_data, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
ax.set_title("model")
ax.set_xlabel('X1 Axis')
ax.set_ylabel('X2 Axis')
ax.set_zlabel('Y Axis')
fig.colorbar(plot, ax=ax, shrink=0.5, aspect=10)

plt.show()