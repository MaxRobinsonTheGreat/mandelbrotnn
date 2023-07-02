import torch
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from src.dataset import MandelbrotDataSet
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt


# Create some random data
# x = (np.random.rand(10000, 2)*2-1)*3
# y = np.sin(x[:, 0] + x[:, 1]).reshape(-1, 1)

dataset = MandelbrotDataSet(10000, max_depth=1500, gpu=True)

x = dataset.inputs.numpy()
y = torch.unsqueeze(dataset.outputs, 1).numpy()

# Define the model
model = SVR(C=50, epsilon=.1, gamma='scale', kernel='rbf')
# model = SVR()

# Define the grid of hyperparameters to search
# hyperparameters = {'C': [0.1, 1, 10, 20], 'epsilon': [0.001, 0.05, 0.1], 'gamma': ['scale', 'auto'], 'kernel': ['poly', 'rbf', 'sigmoid']}

# Use grid search with cross-validation to find the best hyperparameters
# model = GridSearchCV(model, hyperparameters, cv=5, n_jobs=-1)
model.fit(x, y.ravel())

# Print the best hyperparameters
# print("Best Hyperparameters: ", model.best_params_)

# Evaluate the model after training
y_pred = model.predict(x)
print("After training, MSE: ", mean_squared_error(y, y_pred))


# Create a grid of x, y coordinates
x_range = np.linspace(-1, 1, 300)
y_range = np.linspace(-1, 1, 200)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Flatten the grid to 2D and scale it
grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# grid_scaled = scaler.transform(grid)

# Predict the output for each point in the grid
z_pred = model.predict(grid)

# Reshape the predicted output to a 2D grid
z_grid = z_pred.reshape(x_grid.shape)

# z_actual = np.sin(grid[:, 0] + grid[:, 1])

# Reshape the actual output to a 2D grid
# z_grid_actual = z_actual.reshape(x_grid.shape)

# Plot the predicted output as a 2D image
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# axs[0].imshow(z_grid_actual, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
# axs[0].set_title('Actual output')
axs[1].imshow(z_grid, origin='lower', extent=(0, 1, 0, 1), cmap='viridis')
axs[1].set_title('Predicted output')

plt.tight_layout()
plt.show()
