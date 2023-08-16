# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 00:11:16 2023

@author: Nebula
"""
#import numpy as np
import sympy as sp
import math


def R3BP_EOM(t,X):
    #only for 3D case    
    #unpacking state variables
    x = X[0]
    y = X[1]
    z = X[2]
    
    #declaring symbols
    x, y, z, w, mu = sp.symbols('x y z w mu')
    Omega = 0.5*w**2*(x**2+y**2) + mu/math.sqrt(x**2+y**2+z**2)
    
    #partial derivative of potential function
    global Omega_x, Omega_y, Omega_z
    Omega_x = sp.diff(Omega, x)
    print("d(Omega)/dx =", Omega_x)
    Omega_y = sp.diff(Omega, y)
    print("d(Omega)/dy =", Omega_y)
    Omega_z = sp.diff(Omega, z)
    print("d(Omega)/dz =", Omega_z)
    
    xDot = X[3]
    yDot = X[4]
    zDot = X[5]
    
    #equations of motion
    xDDot = 2 * yDot + Omega_x
    yDDot = -2 * xDot + Omega_y
    zDDot = Omega_z

    X_Dot = [xDot, yDot, zDot, xDDot, yDDot, zDDot]


    return X_Dot