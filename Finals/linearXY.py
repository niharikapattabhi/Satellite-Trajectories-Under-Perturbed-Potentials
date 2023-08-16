import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
w = 1
mu = 1
a = 1

x_base = 1
y_base = 1
r_base = np.sqrt(x_base ** 2 + y_base ** 2)

# Initial conditions
x0 = 0
y0 = 2 * a


# ODE system
# x1 = x
# x2 = x_dot
# x3 = y
# x4 = y_dot

def odes(t, z):
    x1, x2, x3, x4 = z

    dxdt = [x2,
            2 * w * x4 - ((mu / r_base ** 3) - w ** 2) * x1,
            x4,
            -2 * w * x2 - ((mu / r_base ** 2) - w ** 2) * x3]

    return dxdt


# Initial state
omega1 = -w - np.sqrt(mu / r_base ** 3)
x_dot0 = -2 * a * omega1
y_dot0 = 0
z0 = [x0, x_dot0, y0, y_dot0]

# Specify time points
t = np.linspace(0, 10, 1000)

# Solve
sol = solve_ivp(odes, [0, 10], z0, t_eval=t)

# Plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[2], label='y(t)')
plt.xlabel('t')
plt.legend()
plt.show()

# Get x and y solution vectors
x = sol.y[0]
y = sol.y[2]

# Plot
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()
