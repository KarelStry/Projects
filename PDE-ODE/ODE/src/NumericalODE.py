# Euler's Method for solving ODEs
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
def euler_method(f, y0, t0, t_end, dt):
    """
    Solves the ODE dy/dt = f(t, y) using Euler's method.
    
    Parameters:
    f : function
        The function defining the ODE.
    y0 : float
        Initial value of y at t0.
    t0 : float
        Initial time.
    t_end : float
        End time for the solution.
    dt : float
        Time step size.
    
    Returns:
    t_values : numpy array
        Array of time values.
    y_values : numpy array
        Array of solution values at each time step.
    """
    t_values = np.arange(t0, t_end + dt, dt)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i - 1] + f(t_values[i - 1], y_values[i - 1]) * dt
    
    return t_values, y_values

# Improved Euler's Method (Heun's Method)
def improved_euler_method(f, y0, t0, t_end, dt):
    """
    Solves the ODE dy/dt = f(t, y) using Improved Euler's method (Heun's method).
    
    Parameters:
    f : function
        The function defining the ODE.
    y0 : float
        Initial value of y at t0.
    t0 : float
        Initial time.
    t_end : float
        End time for the solution.
    dt : float
        Time step size.
    
    Returns:
    t_values : numpy array
        Array of time values.
    y_values : numpy array
        Array of solution values at each time step.
    """
    t_values = np.arange(t0, t_end + dt, dt)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        k1 = f(t_values[i - 1], y_values[i - 1])
        k2 = f(t_values[i - 1] + dt, y_values[i - 1] + k1 * dt)
        y_values[i] = y_values[i - 1] + (k1 + k2) * dt / 2
    
    return t_values, y_values

# Runge-Kutta Method (4th Order)
def runge_kutta_method(f, y0, t0, t_end, dt):
    """
    Solves the ODE dy/dt = f(t, y) using the 4th order Runge-Kutta method.
    
    Parameters:
    f : function
        The function defining the ODE.
    y0 : float
        Initial value of y at t0.
    t0 : float
        Initial time.
    t_end : float
        End time for the solution.
    dt : float
        Time step size.
    
    Returns:
    t_values : numpy array
        Array of time values.
    y_values : numpy array
        Array of solution values at each time step.
    """
    t_values = np.arange(t0, t_end + dt, dt)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        k1 = f(t_values[i - 1], y_values[i - 1])
        k2 = f(t_values[i - 1] + dt / 2, y_values[i - 1] + k1 * dt / 2)
        k3 = f(t_values[i - 1] + dt / 2, y_values[i - 1] + k2 * dt / 2)
        k4 = f(t_values[i - 1] + dt, y_values[i - 1] + k3 * dt)
        y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
    
    return t_values, y_values

t, y = sp.symbols('t y')
# Example ODE: dy/dt = t^2 - y
y0 = 1
t0 = 0
t_end = 2
dt = 0.5
def f(t, y):
    return t**2 - y

# Example usage
t_values, y_values = euler_method(f, y0, t0, t_end, dt)
t_values_improved, y_values_improved = improved_euler_method(f, y0, t0, t_end, dt)
t_values_rk, y_values_rk = runge_kutta_method(f, y0, t0, t_end, dt)

plt.figure()
plt.plot(t_values, y_values, label='Euler Method', color='blue')
plt.plot(t_values_improved, y_values_improved, label='Improved Euler Method', color='orange')
plt.plot(t_values_rk, y_values_rk, label='Runge-Kutta Method', color='green')
plt.show()