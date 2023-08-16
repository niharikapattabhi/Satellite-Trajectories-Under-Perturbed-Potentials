import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import sympy as sp

# Parameters
w = 1
mu = 1

#x_base = 1
#y_base = 1


# Initial conditions
x0 = 1
y0 = 2
z0 = 0


# ODE system
# x1 = x
# x2 = x_dot
# x3 = y
# x4 = y_dot
# x5 = z
# x6 = z_dot





def odes(t, state):
    x1, x2, x3, x4, x5, x6 = state
    r = np.sqrt(x1 ** 2 + x3 ** 2)



    Omega_x = -1.0 * x1 * (1 + 40680622661377.7 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                           x1 ** 2 + x3 ** 2 + x5 ** 2) ** (3 / 2) + 1.0 * x1 + 1.0 * (-81361245322755.4 * x1 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (
                                                                                                   x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 + 40680622661377.7 * (
                                                                                                   0.00324789 * x1 * x5 ** 2 / (
                                                                                                       x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 - 1.0889784e-5 * x1 * (
                                                                                                               x1 ** 2 - x3 ** 2) / (
                                                                                                               x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 + 1.0889784e-5 * x1 / (
                                                                                                               x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                                                                                                   x1 ** 2 + x3 ** 2 + x5 ** 2)) / sp.sqrt(x1 ** 2 + x3 ** 2 + x5 ** 2)
    Omega_y = -1.0 * x3 * (1 + 40680622661377.7 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                           x1 ** 2 + x3 ** 2 + x5 ** 2) ** (3 / 2) + 1.0 * x3 + 1.0 * (-81361245322755.4 * x3 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (
                                                                                                   x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 + 40680622661377.7 * (
                                                                                                   0.00324789 * x3 * x5 ** 2 / (
                                                                                                       x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 - 1.0889784e-5 * x3 * (
                                                                                                               x1 ** 2 - x3 ** 2) / (
                                                                                                               x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 - 1.0889784e-5 * x3 / (
                                                                                                               x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                                                                                                   x1 ** 2 + x3 ** 2 + x5 ** 2)) / sp.sqrt(
        x1 ** 2 + x3 ** 2 + x5 ** 2)
    Omega_z = -1.0 * x5 * (1 + 40680622661377.7 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                           x1 ** 2 + x3 ** 2 + x5 ** 2) ** (3 / 2) + 1.0 * (-81361245322755.4 * x5 * (
                -0.001623945 * x5 ** 2 / (x1 ** 2 + x3 ** 2 + x5 ** 2) + 5.444892e-6 * (x1 ** 2 - x3 ** 2) / (
                    x1 ** 2 + x3 ** 2 + x5 ** 2) + 0.000541315) / (
                                                                                        x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 + 40680622661377.7 * (
                                                                                        0.00324789 * x5 ** 3 / (
                                                                                            x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 - 1.0889784e-5 * x5 * (
                                                                                                    x1 ** 2 - x3 ** 2) / (
                                                                                                    x1 ** 2 + x3 ** 2 + x5 ** 2) ** 2 - 0.00324789 * x5 / (
                                                                                                    x1 ** 2 + x3 ** 2 + x5 ** 2)) / (
                                                                                        x1 ** 2 + x3 ** 2 + x5 ** 2)) / sp.sqrt(
        x1 ** 2 + x3 ** 2 + x5 ** 2)

    dxdt = [x2,
            2 * w * x4 - Omega_x,
            x4,
            -2 * w * x2 - Omega_y,
            x6,
            -Omega_z]

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

# Get x and y solution vectors
x = sol.y[0]
x_dot = sol.y[1]
y = sol.y[2]
y_dot = sol.y[3]
z = sol.y[4]
z_dot = sol.y[5]

# Plot
plt.plot(sol.t, sol.y[0], label='x(t)')
plt.plot(sol.t, sol.y[2], label='y(t)')
plt.plot(sol.t, sol.y[4], label='z(t)')
plt.xlabel('t')
plt.legend()
plt.show()

# Plot
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

plt.plot(x, x_dot)
plt.xlabel('x')
plt.ylabel('x_dot')
plt.title('x vs x_dot')
plt.show()

plt.plot(y, y_dot)
plt.xlabel('y')
plt.ylabel('y_dot')
plt.title('y vs y_dot')
plt.show()

plt.plot(z, z_dot)
plt.xlabel('z')
plt.ylabel('z_dot')
plt.title('z vs z_dot')
plt.show()



