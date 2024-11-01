import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.misc import derivative

# Define the function f(x)
def f(x):
    return x * (x - 1) * (x - 3) * (x + 2)

# Define the derivative using scipy's derivative function
def f_prime(x):
    return derivative(f, x, dx=1e-6)

# Generate x values
x_values = np.linspace(-3, 4, 1000)

# Create a pandas DataFrame
df = pd.DataFrame({
    'x': x_values,
    'f(x)': f(x_values),
    'f\'(x)': f_prime(x_values)
})

# Find critical points using scipy's fsolve to find where f'(x) = 0
initial_guesses = [-2, 0, 2]  # initial guesses for the roots (extrema)
critical_points = fsolve(f_prime, initial_guesses)

# Evaluate function values at critical points
critical_f_values = f(critical_points)

# Add critical points to the DataFrame for plotting
df['critical'] = df['x'].isin(np.round(critical_points, 6))

# Display critical points
for cp, fcp in zip(critical_points, critical_f_values):
    print(f"Critical point at x = {cp:.6f}, f(x) = {fcp:.6f}")

# Plot the function and the critical points
plt.plot(df['x'], df['f(x)'], label='f(x)')
plt.scatter(critical_points, critical_f_values, color='red', label='Extrema', zorder=5)
plt.title('Function and its Extremas')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()