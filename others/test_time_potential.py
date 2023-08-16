# import numpy as np
# from scipy.integrate import solve_ivp
# import matplotlib.pyplot as plt

# # Define constants and parameters
# mu = 1.0       # Example value for mu
# e = 0.1        # Example value for e
# f = 2.0        # Example value for f
# r = 3.0        # Example value for r

# Define the equation to solve
# def potential_equation(t, y):
#     mu, e, f, r = y
#     dydt = [-mu - e * np.sin(f * t) / r,
#             0,  # Change in mu with respect to time is 0
#             0,  # Change in e with respect to time is 0
#             0]  # Change in r with respect to time is 0
#     return dydt
#
# # Set up initial conditions
# initial_conditions = [mu, e, f, r]
# initial_time = 0.0
# final_time = 10.0
# time_points = np.linspace(initial_time, final_time, 1000)
#
# # Solve the equations using solve_ivp
# solution = solve_ivp(potential_equation, (initial_time, final_time), initial_conditions, t_eval=time_points)
#
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(solution.t, solution.y[0], label='mu(t)')
# plt.plot(solution.t, solution.y[1], label='e(t)')
# plt.plot(solution.t, solution.y[2], label='f(t)')
# plt.plot(solution.t, solution.y[3], label='r(t)')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title('Solutions to Time-Dependent Potential Equation')
# plt.legend()
# plt.grid()
# plt.show()
# import numpy as np
# from scipy.integrate import solve_ivp
# import sympy as sp
# import matplotlib.pyplot as plt
#
# # Define symbolic variables
# t_sym = sp.Symbol('t')
# mu_sym, e_sym, f_sym, r_sym = sp.symbols('mu e f r', real=True, positive=True)
#
# # Define the potential equation symbolically
# potential_eq = -mu_sym - e_sym * sp.sin(f_sym * t_sym) / r_sym
#
# # Calculate the derivatives symbolically
# d_mu_dt = sp.diff(potential_eq, mu_sym)
# d_e_dt = sp.diff(potential_eq, e_sym)
# d_f_dt = sp.diff(potential_eq, f_sym)
# d_r_dt = sp.diff(potential_eq, r_sym)
#
# # Convert symbolic derivatives to functions
# d_mu_dt_func = sp.lambdify((t_sym, mu_sym, e_sym, f_sym, r_sym), d_mu_dt, "numpy")
# d_e_dt_func = sp.lambdify((t_sym, mu_sym, e_sym, f_sym, r_sym), d_e_dt, "numpy")
# d_f_dt_func = sp.lambdify((t_sym, mu_sym, e_sym, f_sym, r_sym), d_f_dt, "numpy")
# d_r_dt_func = sp.lambdify((t_sym, mu_sym, e_sym, f_sym, r_sym), d_r_dt, "numpy")
#
# # Define the function for solve_ivp
# def potential_equation(t, y):
#     mu, e, f, r = y
#     dydt = [d_mu_dt_func(t, mu, e, f, r),
#             d_e_dt_func(t, mu, e, f, r),
#             d_f_dt_func(t, mu, e, f, r),
#             d_r_dt_func(t, mu, e, f, r)]
#     return dydt
#
# # Set up initial conditions
# initial_conditions = [1.0, 0.1, 2.0, 3.0]  # Example initial conditions
# initial_time = 0.0
# final_time = 10.0
# time_points = np.linspace(initial_time, final_time, 1000)
#
# # Solve the equations using solve_ivp
# solution = solve_ivp(potential_equation, (initial_time, final_time), initial_conditions, t_eval=time_points)
#
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(solution.t, solution.y[0], label='mu(t)')
# plt.plot(solution.t, solution.y[1], label='e(t)')
# plt.plot(solution.t, solution.y[2], label='f(t)')
# plt.plot(solution.t, solution.y[3], label='r(t)')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.title('Solutions to Time-Dependent Potential Equation')
# plt.legend()
# plt.grid()
# plt.show()
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, sin, sqrt

# Define symbols
t_sym = symbols('t')
mu_sym, e_sym, f_sym, x_sym, y_sym, z_sym = symbols('mu e f x y z')

# Define potential equation and its derivatives
r_sym = sqrt(x_sym ** 2 + y_sym ** 2 + z_sym ** 2)
potential = -mu_sym - e_sym * sin(f_sym * t_sym) / r_sym


# Define the equation to solve
def potential_equation(t, y):
    mu, e, f, x, y, z = y
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    dxdt = 0  # Change in x with respect to time is 0
    dydt = 0  # Change in y with respect to time is 0
    dzdt = 0  # Change in z with respect to time is 0
    dmudt = 0  # Change in mu with respect to time is 0
    dedt = 0  # Change in e with respect to time is 0
    dfdt = 0  # Change in f with respect to time is 0

    return [dmudt, dedt, dfdt, dxdt, dydt, dzdt]


# Replace with actual initial conditions
mu_initial = 1.0
e_initial = 0.1
f_initial = 2.0
x_initial = 0.0
y_initial = 0.0
z_initial = 0.0
initial_conditions = [mu_initial, e_initial, f_initial, x_initial, y_initial, z_initial]

initial_time = 0.0
final_time = 10.0
time_points = np.linspace(initial_time, final_time, 1000)

# Solve the equations using solve_ivp
solution = solve_ivp(potential_equation, (initial_time, final_time), initial_conditions, t_eval=time_points)


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label='mu(t)')
plt.plot(solution.t, solution.y[1], label='e(t)')
plt.plot(solution.t, solution.y[2], label='f(t)')
plt.plot(solution.t, solution.y[3], label='x(t)')
plt.plot(solution.t, solution.y[4], label='y(t)')
plt.plot(solution.t, solution.y[5], label='z(t)')
plt.xlabel('Time')
plt.ylabel('Values')
plt.title('Solutions to Time-Dependent Potential Equation')
plt.legend()
plt.grid()
plt.show()

