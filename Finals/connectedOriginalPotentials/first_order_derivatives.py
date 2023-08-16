import sympy as sp
from get_potential import x, y, z


def first_order_derivatives(Omega):
    print("\nThe first order partial derivatives are: \n")
    Omega_x = sp.diff(Omega, x)
    print("Omega_x =", Omega_x)
    Omega_y = sp.diff(Omega, y)
    print("Omega_y =", Omega_y)
    Omega_z = sp.diff(Omega, z)
    print("Omega_z =", Omega_z)
    return Omega_x, Omega_y, Omega_z
