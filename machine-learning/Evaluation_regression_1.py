# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,root_mean_squared_error, r2_score, mean_absolute_error

# Generate synthetic regression data
X,y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# visualize the data
plt.scatter(X, y, color='blue')
plt.title('Synthetic Regression Data')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
rmse = root_mean_squared_error(y_test, y_pred)
print(f'Root Mean Squared Error: {rmse}')
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')
r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')

# Visualize the predictions
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.title('Linear Regression Predictions')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()