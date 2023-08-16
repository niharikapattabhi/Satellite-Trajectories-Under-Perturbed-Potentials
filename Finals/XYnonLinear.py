import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
w = 1
mu = 1


#x_base = 1
#y_base = 1


# Initial conditions
x0 = 1
y0 = -2
z0 = 0


# ODE system
# x1 = x
# x2 = x_dot
# x3 = y
# x4 = y_dot

def odes(t, state):
    x1, x2, x3, x4, x5, x6 = state
    r = np.sqrt(x1 ** 2 + x3 ** 2 + x5 ** 2)

    dxdt = [x2,
            2 * w * x4 - ((mu / r ** 3) - w ** 2) * x1,
            x4,
            -2 * w * x2 - ((mu / r ** 2) - w ** 2) * x3,
            x6,
            - ((mu / r ** 2) - w ** 2) * x5]

    return dxdt


# Initial state
x_dot0 = 1
y_dot0 = -1
z_dot0 = 0
state0 = [x0, x_dot0, y0, y_dot0, z0, z_dot0]

# Specify time points
t = np.linspace(0, 100, 5000)

# Solve
sol = solve_ivp(odes, [0, 100], state0, t_eval=t)

# Plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[2], label='y(t)')
plt.plot(sol.t, sol.y[4], label='z(t)')
plt.xlabel('t')
plt.legend()
plt.show()

# Get x and y solution vectors
x = sol.y[0]
y = sol.y[2]
z = sol.y[4]

# Plot
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()

