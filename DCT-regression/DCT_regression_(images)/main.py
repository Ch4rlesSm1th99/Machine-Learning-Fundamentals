"""
DCT Regression, which utilizes the Discrete Cosine Transform (DCT), can be thought of as an adaptation of linear
regression for image or signal data. Unlike traditional linear regression, where we might use a set of independent
variables or features to predict a dependent variable, DCT Regression uses a series of trigonometric basis functions to
represent or approximate an image.
"""

# DCT regression for images
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

# Creating the original image: a white square on a black background
im = np.zeros((64, 64))
im[30:50, 30:50] = 1

# Set up parameters for the Discrete Cosine Transform (DCT) basis images
M = 32  # Number of basis images per dimension
nr, nc = im.shape
x1 = np.arange(nr) / nr  # Normalized row coordinates
x2 = np.arange(nc) / nc  # Normalized column coordinates
I = np.arange(M).reshape(1, 1, -1, 1)
J = np.arange(M).reshape(1, 1, 1, -1)
X1, X2 = np.meshgrid(x1, x2)  # Create meshgrid for 2D basis computation
X1t = X1.reshape(nr, nc, 1, 1)
X2t = X2.reshape(nr, nc, 1, 1)

# Compute the DCT basis images
basis = np.cos(pi * X1t * I) * np.cos(pi * X2t * J)

# Visualize DCT basis images in a grid
plt.figure(figsize=(10, 10))
k = 0
for i in range(M):
    for j in range(M):
        k += 1
        plt.subplot(M, M, k)
        plt.imshow(basis[:, :, i, j], cmap='gray')
        plt.axis("off")

# Prepare the matrix X using the DCT basis images to compute the regression
X = basis.reshape(nr * nc, M ** 2)
# Compute the regression coefficients theta
theta = np.linalg.solve(X.T @ X, X.T @ im.reshape(-1, 1))
# Get the DCT regression result
im_hat = (X @ theta).reshape(nr, nc)

# Display original and DCT regression side-by-side
plt.figure(figsize=(10, 5))
# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Original Image')
plt.axis("off")
# Display the DCT regression
plt.subplot(1, 2, 2)
plt.imshow(im_hat, cmap='gray')
plt.title('DCT Regression')
plt.axis("off")

# Adjust the spacing and display the plots
plt.tight_layout()
plt.show()
