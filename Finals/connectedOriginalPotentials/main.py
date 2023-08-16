# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 02:26:36 2023

@author: Nebula
"""
#library imports
import matplotlib.pyplot as plt

#connectedOriginalPotentials file imports
import get_potential
import first_order_derivatives
# import df_matrix
import numerical_integration


def main():
    mu = float(input("Enter mu: "))
    w = float(input("Enter w: "))

    # Initial state
    def get_initial():
        initial_x = float(input("Enter initial x: "))
        initial_y = float(input("Enter initial y: "))
        initial_z = float(input("Enter initial z: "))
        return initial_x, initial_y, initial_z

    initial_x, initial_y, initial_z = get_initial()

    # constants
    alpha = 6378136.3
    C_20 = -0.1082630e-2
    C_22 = 0.1814964e-5

    Omega = get_potential.get_potential(mu, w, alpha, C_20, C_22)

    # Pass initial state to get derivatives
    Omega_x, Omega_y, Omega_z = first_order_derivatives.first_order_derivatives(Omega)

    # Df = df_matrix.df_matrix(Omega_x, Omega_y, Omega_z, w)

    # Integrate ODEs
    t, x_t, y_t, z_t = numerical_integration.numerical_integration(Omega_x, Omega_y, Omega_z, initial_x, initial_y, initial_z, w)

   # # Plot x_t vs t
   #  plt.figure()
   #  plt.plot(t, x_t)
   #  plt.xlabel('Time (t)')
   #  plt.ylabel('x(t)')
   #  plt.title('x vs t')
   #  plt.show()
   #  plt.close()
   #
   #  # Plot y_t vs t
   #  plt.figure()
   #  plt.plot(t, y_t)
   #  plt.xlabel('Time (t)')
   #  plt.ylabel('y(t)')
   #  plt.title('y vs t')
   #  plt.show()
   #  plt.close()

    # 3D plot
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(x_t, y_t, z_t)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title('3D Trajectory')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
