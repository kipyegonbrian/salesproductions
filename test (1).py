# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some sample data for a simple linear regression model
# Let's create data for a linear relation: y = 3x + 7 with some noise

np.random.seed(0)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random data points for X
y = 7 + 3 * X + np.random.randn(100, 1)  # y = 3x + 7 + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Display the model's parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

# Plot the results
plt.scatter(X_test, y_test, color="blue", label="Actual values")
plt.plot(X_test, y_pred, color="red", label="Predicted values")
plt.xlabel("X values")
plt.ylabel("y values")
plt.title("Linear Regression Prediction")
plt.legend()
plt.show()
