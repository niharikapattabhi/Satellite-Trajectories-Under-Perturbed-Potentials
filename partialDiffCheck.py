# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:50:38 2023

@author: Nebula
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sympy import *


'''x = symbols('x')
f = 2*x**2+5
df = diff(f,x)
print(df)'''

x,y,z,w,mu = symbols('x y z w mu')


Omega = 0.5*w**2*(x**2+y**2) + mu/sqrt(x**2+y**2+z**2)
dOmega_x = diff(Omega, x)
print("ux = ", dOmega_x)
print()

dOmega_y = diff(Omega, y)
print("uy = ", dOmega_y)
print()

dOmega_z = diff(Omega, z)
print("uz = ", dOmega_z)
print()

Uxx = diff(dOmega_x, x)
print("uxx = ", Uxx)
print()

Uxy = diff(dOmega_x, y)
print("uxy = ", Uxy)
print()

Uxz = diff(dOmega_x, z)
print("uxz = ", Uxz)
print()

Uyx = diff(dOmega_y, x)
print("uyx = ", Uyx)
print()

Uyy = diff(dOmega_y, y)
print("uyy = ", Uyy)
print()

Uyz = diff(dOmega_y, z)
print("uyz = ", Uyz)
print()

Uzx = diff(dOmega_z, x)
print("uzx = ", Uzx)
print()

Uzy = diff(dOmega_z, y)
print("uzy = ", Uzy)
print()

Uzz = diff(dOmega_z, z)
print("uzz = ", Uzz)



'''# Define the equations of motion in the second-order form
def equations_of_motion(t, u):
    x, y, x_dot, y_dot = u[0], u[1], u[2], u[3]
    omega = 1.0  # Replace with the desired value of omega
    mu = 1.0  # Replace with the desired value of mu

    x_double_dot = omega ** 2 * x - 2 * omega * y_dot - mu * x / (x ** 2 + y ** 2) ** (3 / 2)
    y_double_dot = omega ** 2 * y + 2 * omega * x_dot - mu * y / (x ** 2 + y ** 2) ** (3 / 2)

    return [x_dot, y_dot, x_double_dot, y_double_dot]

# Set initial conditions
initial_conditions = [1.0, 1.0, 1.0, 1.0]  # Replace with your desired initial conditions

# Set the integration time span
t_span = (0.0, 10.0)  # Replace with the desired time span

# Integrate the equations of motion
solution = solve_ivp(equations_of_motion, t_span, initial_conditions)

# Retrieve the time points and solution values
t = solution.t
x = solution.y[0]
y = solution.y[1]
x_dot = solution.y[2]
y_dot = solution.y[3]

# Do further processing or plotting with the obtained solution values
# Plotting the solution
plt.figure(figsize=(8, 6))
plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position vs. Time')
plt.legend()
plt.grid(True)
plt.show()'''
