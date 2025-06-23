import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Spring mass-damper with homogeneous boundary conditions
# Spring mass-damper system parameters
m = 1.0  # Mass (kg)
k = 0.5  # Spring constant (N/m)
c = 0.5  # Damping coefficient (N·s/m)

# Set up characteristic equation coefficients
a = c/m
b = k/m

# Define the symbolic variable
r = sp.symbols('r')

char_eq = r**2 + a*r + b

# Solve the characteristic equation
roots = sp.solve(char_eq, r)

damping_ratio = c/(2 * np.sqrt(m * k))
natural_frequency = np.sqrt(k/m)

t = sp.symbols('t')

if damping_ratio < 1:
    # Underdamped case
    omega_d = natural_frequency * np.sqrt(1 - damping_ratio**2)
    t = np.linspace(0, 10, 1000)
    response = np.exp(-damping_ratio * natural_frequency * t) * np.cos(omega_d * t)
    plt.plot(t, response, label='Underdamped Response')
elif damping_ratio == 1:
    # Critically damped case
    response = np.exp(-natural_frequency * t) * (1 + natural_frequency * t)
    plt.plot(t, response, label='Critically Damped Response')
elif damping_ratio > 1:
    # Overdamped case
    r1, r2 = roots
    response = np.exp(r1 * t) + np.exp(r2 * t)
    plt.plot(t, response, label='Overdamped Response')

plt.title('Spring Mass-Damper System Response')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.legend()
plt.grid()
plt.show()
