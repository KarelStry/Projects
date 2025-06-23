import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 0.1      # Reduced time
alpha = 1.0

N = 50       # Reduced spatial points
dx = L / N

M = 1000      # Reduced time steps
dt = T / M

x = np.linspace(0, L, N+1)
r = alpha * dt / dx**2

if r > 0.5:
    print("Warning: CFL condition violated. Reduce dt or increase dx.")

u_initial = np.sin(np.pi * x)
u = u_initial.copy()
u_new = np.zeros_like(u)

u[0] = 0
u[-1] = 0

for n in range(M):
    for i in range(1, N):
        u_new[i] = u[i] + r * (u[i+1] - 2*u[i] + u[i-1])
    u_new[0] = 0
    u_new[-1] = 0
    u[:] = u_new[:]

print("Max u after simulation:", np.max(u))

plt.plot(x, u_initial, label='Initial condition')
plt.plot(x, u, label='t = {:.2f}'.format(T))
plt.title('1D Heat Equation Solution')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.legend()
plt.grid()
plt.show()
