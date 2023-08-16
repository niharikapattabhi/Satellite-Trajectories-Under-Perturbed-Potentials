import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#values
mu = 1
w = 1
e = 1
f = 1

# Symbols
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
T = sp.symbols('T')

ft = f * T

# Potential
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
potential = -(mu + e * sp.sin(ft)) / r + 0.5 * w ** 2 * (x ** 2 + y ** 2)


# Numeric potential
def get_numeric_potential(t, x_val, y_val, z_val):
    return potential.subs({T: t, x: x_val, y: y_val, z: z_val})


# Equations of motion
def equations_of_motion(t, state, w):
    x_val, y_val, z_val = state[:3]
    U = get_numeric_potential(t, x_val, y_val, z_val)
    dUdx = sp.diff(U, x)
    dUdy = sp.diff(U, y)
    dUdz = sp.diff(U, z)
    xDot, yDot, zDot = state[3:]
    xDDot = 2 * w * yDot - dUdx
    yDDot = -2 * w * xDot - dUdy
    zDDot = -dUdz
    return [xDot, yDot, zDot, xDDot, yDDot, zDDot]


# Initial conditions
x0 = 1
y0 = 2
z0 = 0
xDot0 = 1
yDot0 = -1
zDot0 = 0
t0 = 0
tf = 100
t = np.linspace(0, 100, 5000)
initial_conditions = [x0, y0, z0, xDot0, yDot0, zDot0]

# Integrate
solution = solve_ivp(equations_of_motion, (t0, tf), initial_conditions, t_eval=t, args=(w,))

# Plot trajectories
t = solution.t
x_vals = solution.y[0]
y_vals = solution.y[1]
z_vals = solution.y[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals)
plt.show()

''''# Animate potential
X, Y = np.meshgrid(np.linspace(-5, 5, 30), np.linspace(-5, 5, 30))

for t_val in t:
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = get_numeric_potential(t_val, X[i, j], Y[i, j], 0)
    ax.clear()  # Clear previous frame
    ax.plot_surface(X, Y, Z)
    plt.pause(0.1)

plt.show()'''

# Plotting x, y, and z against time
plt.figure(figsize=(10, 6))
plt.plot(t, x_vals, label='x')
plt.plot(t, y_vals, label='y')
plt.plot(t, z_vals, label='z')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Particle Trajectories')
plt.legend()
plt.grid(True)
plt.show()
