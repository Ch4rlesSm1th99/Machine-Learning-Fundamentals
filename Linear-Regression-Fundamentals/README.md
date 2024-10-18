# Linear-Regression-Fundamentals
An explanation of the most fundamental underlying steps for implementing any kind of linear regression model on a set of data. Here is a methodical explanation of the script.

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

