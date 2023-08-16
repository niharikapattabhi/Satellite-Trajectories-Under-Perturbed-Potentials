# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 12:47:34 2023

@author: Nebula
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def L0U(U, c, k, cp):
    return (0.5 * np.fft.fft(U**2) + (-c + cp(k)) * np.fft.fft(U))

def get_per_sol(U, c, k, cp, x):
    opts = {'xtol': 1e-8, 'maxfev': int(1e4), 'factor': 0.1}
    U_real = np.real(U)
    U_imag = np.imag(U)
    
    def func(U):
        U_complex = U_real + 1j * U
        return L0U(U_complex, c, k, cp)
    
    U_real, = fsolve(func, U_imag, **opts)
    U = U_real + 1j * U_imag
    
    return U

L = 4 * np.pi
N = 2 ** 6
plotFlag = 1
x = np.linspace(-L/2, L/2, N, endpoint=False)
k = 2 * np.pi / L * np.fft.fftshift(np.arange(-N/2, N/2))
sigma = -1

def cp(k):
    return (-sigma * k**2 + k**4)

V_init = cp(2 * np.pi / L)
U = np.cos(2 * np.pi / L * x)

for velPert in np.arange(0.05, 20.05, 0.05):
    c = V_init + velPert
    U = get_per_sol(U, c, k, cp, x)
    
    if plotFlag:
        plt.subplot(2, 2, [1, 2])
        plt.plot(x, U)
        plt.xlabel('x')
        plt.ylabel('U')
        plt.title(f'solution velocity c = {c:.3f}')
        plt.axis('tight')
        
        plt.subplot(2, 2, 3)
        plt.plot(c, np.sum(np.fft.fft(abs(U / N) ** 2)), 'k.')
        plt.xlabel('$c$', interpreter='latex')
        plt.ylabel('$\|U\|^2$', interpreter='latex')
        
        plt.subplot(2, 2, 4)
        plt.semilogy(np.fft.fftshift(k), np.fft.fftshift(abs(np.fft.fft(U))) / len(U))
        plt.xlabel('$k$', interpreter='latex')
        plt.ylabel('$|\hat{U}|^2$', interpreter='latex')
        plt.xlim([0, np.max(k)])
        plt.ylim([1e-20, 10])
        
        plt.draw()
        plt.show()


