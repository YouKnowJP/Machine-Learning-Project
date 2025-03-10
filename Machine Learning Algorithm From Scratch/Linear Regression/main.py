# main.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('score_updated.csv')

# Display first few rows
print(data.head())

# Loss function (Mean Squared Error)
def loss_function(m, b, points):
    x = points['Hours'].values
    y = points['Scores'].values
    return np.mean((y - (m * x + b)) ** 2)

# Gradient descent optimization
def gradient_descent(m_now, b_now, points, learning_rate):
    x = points['Hours'].values
    y = points['Scores'].values
    n = len(points)

    # Compute gradients
    y_pred = m_now * x + b_now
    m_gradient = (-2/n) * np.sum(x * (y - y_pred))
    b_gradient = (-2/n) * np.sum(y - y_pred)

    # Update parameters
    m_next = m_now - learning_rate * m_gradient
    b_next = b_now - learning_rate * b_gradient
    return m_next, b_next

# Initialize parameters
m, b = 0, 0
learning_rate = 0.01
epochs = 1000

# Training loop
for epoch in range(epochs):
    m, b = gradient_descent(m, b, data, learning_rate)
    if epoch % 50 == 0:
        loss = loss_function(m, b, data)
        print(f"Epoch {epoch}: m = {m:.4f}, b = {b:.4f}, Loss = {loss:.4f}")

# Plot results
plt.scatter(data['Hours'], data['Scores'], color='red', label="Actual Data")
x_range = np.linspace(data['Hours'].min(), data['Hours'].max(), 100)
y_range = m * x_range + b
plt.plot(x_range, y_range, color='blue', label="Fitted Line")

# Labels and legend
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.legend()
plt.show()
