# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 10:50:38 2023

@author: Nebula
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy.integrate import solve_ivp


# initial guess
def initial_condition():
    global mu, w
    mu = float(input("Enter mu value: "))  # Define your value of mu
    w = float(input("Enter w value: "))  # Define your value of w
    initial_x = float(input("Initial value of x: "))
    initial_y = float(input("Initial value of y: "))
    initial_z = float(input("Initial value of z: "))
    initial_xDot = float(input("Initial value of xDot: "))
    initial_yDot = float(input("Initial value of yDot: "))
    initial_zDot = float(input("Initial value of zDot: "))
    initial_state = [float(initial_x), float(initial_y), float(initial_z),
                     float(initial_xDot), float(initial_yDot), float(initial_zDot)]

    start_time = float(input("Input start time for t_span: "))
    end_time = float(input("Input end time for t_span: "))
    num_points = int(input("Enter number of points: "))
    time_span = (start_time, end_time)

    result = solve_ivp(fun=lambda t, y: equations_of_motion(t, y, mu, w),
                       t_span=time_span,
                       y0=initial_state,
                       method='RK45',  # You can choose a different method if needed
                       t_eval=np.linspace(start_time, end_time, num_points)
                       )

    # Extract the solution
    t = result.t
    print(t)
    x, y, z, xDot, yDot, zDot = result.y

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory of (x, y) over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


# equations of motion
def equations_of_motion(t, state, mu, w):
    x, y, z, xDot, yDot, zDot = state

    # Calculate the potential and its derivatives at the current state

    xDDot = 2 * yDot + Omega_x
    yDDot = -2 * xDot + Omega_y
    zDDot = Omega_z

    return [xDot, yDot, zDot, xDDot, yDDot, zDDot]
