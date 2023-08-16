from scipy.integrate import solve_ivp
import equations_of_motion
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

"""Numerically integrate equations of motion.

Args:
  Omega_x, Omega_y, Omega_z: Potential derivatives
  initial_x, initial_y, initial_z: Initial state

Returns:
  t, x, y, z: Result arrays 
"""


def numerical_integration(Omega_x, Omega_y, Omega_z, initial_x, initial_y, initial_z, w):
    x, y, z = sp.symbols('x y z')
    initial_xDot = float(input("Initial value of xDot: "))
    initial_yDot = float(input("Initial value of yDot: "))
    initial_zDot = float(input("Initial value of zDot: "))
    initial_state = [float(initial_x), float(initial_y), float(initial_z),
                     float(initial_xDot), float(initial_yDot), float(initial_zDot)]

    # Substitute values
    # numerical potential derivatives
    Omega_x = Omega_x.subs({x: initial_x, y: initial_y, z: initial_z})
    Omega_y = Omega_y.subs({x: initial_x, y: initial_y, z: initial_z})
    Omega_z = Omega_z.subs({x: initial_x, y: initial_y, z: initial_z})

    start_time = float(input("start time for t_span: "))
    end_time = float(input("end time for t_span: "))
    num_points = int(input("number of points: "))
    time_span = (start_time, end_time)

    result = solve_ivp(fun=lambda t, y: equations_of_motion.equations_of_motion(t, y, Omega_x, Omega_y, Omega_z, w),
                       t_span=time_span,
                       y0=initial_state,
                       method='RK45',  # You can choose a different method if needed
                       t_eval=np.linspace(start_time, end_time, num_points)
                       )

    # Extract the solution
    t = result.t
    x_traj = result.y[0]
    y_traj = result.y[1]
    z_traj = result.y[2]
    print(t)
    # Plot
    plt.figure()
    plt.plot(t, x_traj, t, y_traj)
    plt.xlabel('t')
    plt.ylabel('x, y')
    plt.title('Trajectory')
    plt.legend(['x', 'y'])
    plt.show()

    return t, x_traj, y_traj, z_traj
