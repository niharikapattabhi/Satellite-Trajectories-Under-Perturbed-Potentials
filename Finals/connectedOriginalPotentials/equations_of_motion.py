import first_order_derivatives


def equations_of_motion(t, state, Omega_x, Omega_y, Omega_z, w):
    x, y, z, xDot, yDot, zDot = state

    a = xDot
    b = yDot
    c = zDot

    # Calculate the potential and its derivatives at the current state
    aDot = 2 * yDot * w + Omega_x
    bDot = -2 * xDot * w + Omega_y
    cDot = Omega_z

    return [a, b, c, aDot, bDot, cDot]
