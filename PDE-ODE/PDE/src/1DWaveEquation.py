import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 2.0         # Length of string
T = 2.0         # Total time
c = 1.0         # Wave speed
nx = 100        # Number of spatial points
nt = 100        # Number of time steps
dx = L / (nx - 1)
dt = T / nt
x = np.linspace(0, L, nx)

# CFL condition (must be <= 1 for stability)
C = c * dt / dx
assert C <= 1, "CFL condition not met"

# Initial conditions
u = np.zeros((nt, nx))
u[0] = np.sin(np.pi * x)  # Initial displacement
u[1, 1:-1] = u[0, 1:-1] + 0.5 * C**2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])  # First time step

# Time stepping
for n in range(1, nt - 1):
    u[n+1, 1:-1] = (2 * u[n, 1:-1] - u[n-1, 1:-1] +
                    C**2 * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))

# Animation
fig, ax = plt.subplots()
line, = ax.plot(x, u[0])
ax.set_ylim(-1.2, 1.2)
def update(frame):
    line.set_ydata(u[frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=nt, interval=30)
plt.show()
