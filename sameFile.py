# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 02:26:36 2023

@author: Nebula
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

# declaring symbols
x, y, z, w, mu, alpha, C_20, C_22 = sp.symbols('x y z w mu alpha C_20 C_22')
# variable definitions or constant values
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)


# potential equation
def get_potential():
    # potential equations user input
    # mu is gravitational constant, r is radial distance
    # alpha is equatorial radius, C_20,22 are harmonic coefficients
    # Omega is effective potential function
    global Omega, mu, w, initial_x, initial_y, initial_z
    mu = float(input("Enter mu value: "))  # Define your value of mu
    w = float(input("Enter w value: "))  # Define your value of w
    initial_x = float(input("Initial value of x: "))
    initial_y = float(input("Initial value of y: "))
    initial_z = float(input("Initial value of z: "))
    alpha = 6378136.3
    C_20 = -0.1082630e-2
    C_22 = 0.1814964e-5

    print("Choose a potential equation")
    potential = input('''Available equations:
              1) mu/r
              2) (mu/r)*(1 + (alpha/r)**2 {3*C_22*((x**2-y**2)/r**2) - 0.5*C_20*(1-3(z**2/r**2))})
          ''')

    # potential equations switch menu
    if potential == '1':
        Omega = 0.5 * w ** 2 * (x ** 2 + y ** 2) + mu / r
        Omega = Omega.subs({mu: mu, x: initial_x, y: initial_y, z: initial_z})
    elif potential == '2':
        Omega = 0.5 * w ** 2 * (x ** 2 + y ** 2) + (mu / r) * (1 + (alpha / r) ** 2 * (
                3 * C_22 * ((x ** 2 - y ** 2) / r ** 2) - 0.5 * C_20 * (1 - 3 * (z ** 2 / r ** 2))))
        Omega = Omega.subs({mu: mu, alpha: alpha, C_20: C_20, C_22: C_22,
                            x: initial_x, y: initial_y, z: initial_z})
    else:
        print("Choose either 1 or 2")


# partial derivative calculation
def first_order_derivatives():
    global Omega_x, Omega_y, Omega_z
    print("\nThe first order partial derivatives are: \n")
    Omega_x = sp.diff(Omega, x)
    print("Omega_x =", Omega_x)
    Omega_y = sp.diff(Omega, y)
    print("Omega_y =", Omega_y)
    Omega_z = sp.diff(Omega, z)
    print("Omega_z =", Omega_z)


# Df matrix calculation
def df_matrix():
    print("\nThe second order partial differentials are: \n")
    Omega_xx = sp.diff(Omega_x, x)
    print("Omega_xx =", Omega_xx)
    Omega_yy = sp.diff(Omega_y, y)
    print("Omega_yy =", Omega_yy)
    Omega_zz = sp.diff(Omega_z, z)
    print("Omega_zz =", Omega_zz)

    print("\nThe second order cross partial differentials are: \n")
    Omega_xy = sp.diff(Omega_x, y)
    print("Omega_xy =", Omega_xy)
    Omega_xz = sp.diff(Omega_x, z)
    print("Omega_xz =", Omega_xz)

    Omega_yx = sp.diff(Omega_y, x)
    print("Omega_yx =", Omega_yx)
    Omega_yz = sp.diff(Omega_y, z)
    print("Omega_yz =", Omega_yz)

    Omega_zx = sp.diff(Omega_z, x)
    print("Omega_zx =", Omega_zx)
    Omega_zy = sp.diff(Omega_z, y)
    print("Omega_zy =", Omega_zy)

    I = sp.eye(3)
    zer = sp.zeros(3, 3)
    UXX = sp.Matrix([[Omega_xx, Omega_xy, Omega_xz],
                     [Omega_yx, Omega_yy, Omega_yz],
                     [Omega_zx, Omega_zy, Omega_zz]])
    sig = sp.Matrix([[0, 2 * w, 0], [-2 * w, 0, 0], [0, 0, 0]])

    Df = sp.BlockMatrix([[zer, I], [UXX, sig]])
    print(Df)


# initial guess
def initial_condition(x, y, z, Omega_x, Omega_y, Omega_z):


    initial_xDot = float(input("Initial value of xDot: "))
    initial_yDot = float(input("Initial value of yDot: "))
    initial_zDot = float(input("Initial value of zDot: "))
    initial_state = [float(initial_x), float(initial_y), float(initial_z),
                     float(initial_xDot), float(initial_yDot), float(initial_zDot)]

    start_time = float(input("Input start time for t_span: "))
    end_time = float(input("Input end time for t_span: "))
    num_points = int(input("Enter number of points: "))
    time_span = (start_time, end_time)

    # Substitute values
    Omega_x.subs({x: 1.0, y: 2.0, z: 3.0})
    Omega_y.subs({x: 1.0, y: 2.0, z: 3.0})
    Omega_z.subs({x: 1.0, y: 2.0, z: 3.0})


    result = solve_ivp(fun=lambda t, y: equations_of_motion(t, y),
                       t_span=time_span,
                       y0=initial_state,
                       method='RK45',  # You can choose a different method if needed
                       t_eval=np.linspace(start_time, end_time, num_points)
                       )

    # Extract the solution
    global t
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
def equations_of_motion(t, state):
    x, y, z, xDot, yDot, zDot = state

    # Calculate the potential and its derivatives at the current state
    xDDot = 2 * yDot + Omega_x
    yDDot = -2 * xDot + Omega_y
    zDDot = Omega_z

    return [xDot, yDot, zDot, xDDot, yDDot, zDDot]


# main script
def main():
    get_potential()
    first_order_derivatives()
    df_matrix()

    initial_condition(x, y, z, Omega_x, Omega_y, Omega_z)


if __name__ == "__main__":
    main()
