import sympy as sp
from get_potential import x, y, z


def df_matrix(Omega_x, Omega_y, Omega_z, w):
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
    return Df
