# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:16:13 2023

@author: Nebula
"""
import numpy as np
import sympy as sp
from R3BP_EOM import *

def DfMatrix3D (X):
    '''r2 = (x[0] + 1) ** 2 + x[1] ** 2 + x[2] ** 2 #dist from larger mass
    R2 = (x[0] - 1) ** 2 + x[1] ** 2 + x[2] ** 2 #dis from smaller mass

    r3 = r2 ** 1.5
    r5 = r2 ** 2.5
    R3 = R2 ** 1.5
    R5 = R2 ** 2.5'''
    
    Omega_xx = sp.diff(Omega_x, x)
    Omega_yy = sp.diff(Omega_y, y)
    Omega_zz = sp.diff(Omega_z, z)
    
    Omega_xy = sp.diff(Omega_x, y)
    Omega_xz = sp.diff(Omega_x, z)
    
    Omega_yx = sp.diff(Omega_y, x)
    Omega_yz = sp.diff(Omega_y, z)
    
    Omega_zx = sp.diff(Omega_z, x)
    Omega_zy = sp.diff(Omega_z, y)
    
    I = sp.eye(3)
    zer = sp.zeros(3, 3)
    UXX = sp.Matrix([[Omega_xx, Omega_xy, Omega_xz], 
                     [Omega_yx, Omega_yy, Omega_yz], 
                     [Omega_zx, Omega_zy, Omega_zz]])
    sig = sp.Matrix([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])
    
    Df = sp.BlockMatrix([[zer, I], [UXX, sig]])
    
    print(Df)
    
    return Df