import sympy as sp


x, y, z = sp.symbols('x y z')
r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)


def get_potential(mu, w, alpha, C_20, C_22):
    # Omega is effective potential function

    print("Choose a potential equation")
    potential = input('''Available equations:
              1) -mu/r
              2) (-mu/r)*(1 + (alpha/r)**2 {3*C_22*((x**2-y**2)/r**2) - 0.5*C_20*(1-3(z**2/r**2))})
          ''')

    # potential equations switch menu
    if potential == '1':
        Omega = 0.5 * w ** 2 * (x ** 2 + y ** 2) + mu / r
    elif potential == '2':
        Omega = (0.5 * w ** 2 * (x ** 2 + y ** 2) + (mu / r) * (1 + (alpha / r) ** 2 * (
                    3 * C_22 * ((x ** 2 - y ** 2) / r ** 2) - 0.5 * C_20 * (1 - 3 * (z ** 2 / r ** 2)))))
    else:
        print("Choose either 1 or 2")
    return Omega
