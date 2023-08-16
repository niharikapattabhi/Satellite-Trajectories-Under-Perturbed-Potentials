# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:07:51 2023

@author: Nebula

"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def myf(t, y):
    yprime = -y -5*np.exp(-t)*np.sin(5*t)
    
    return yprime

tspan = (0, 3)
yzero = [1]
sol = solve_ivp(myf, tspan, yzero, method='RK45', t_eval=np.linspace(0, 3, 25))

plt.plot(sol.t, sol.y[0], '*--')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('ODE Solution using solve_ivp with RK45')
plt.show()
#----------------------------------------------------------------------------------------------------------------
def pend(y, t):
    y1, y2 = y
    dy1dt = y2
    dy2dt = -np.sin(y1)
    return [dy1dt, dy2dt]

t_span = (0,10)
yazero = [1, 1]
ybzero = [-5, 2]
yczero = [5, -2]

tspan = np.array([0, 10])
yazero = np.array([1, 1])
ybzero = np.array([-5, 2])
yczero = np.array([5, -2])

ta = np.linspace(tspan[0], tspan[1], 100)
ya = odeint(pend, yazero, ta)

tb = np.linspace(tspan[0], tspan[1], 100)
yb = odeint(pend, ybzero, tb)

tc = np.linspace(tspan[0], tspan[1], 100)
yc = odeint(pend, yczero, tc)

y1, y2 = np.meshgrid(np.arange(-5, 5, 0.5), np.arange(-3, 3, 0.5))
Dy1Dt = y2
Dy2Dt = -np.sin(y1)

plt.quiver(y1, y2, Dy1Dt, Dy2Dt)
plt.plot(ya[:, 0], ya[:, 1], label='ya')
plt.plot(yb[:, 0], yb[:, 1], label='yb')
plt.plot(yc[:, 0], yc[:, 1], label='yc')
plt.axis('equal')
plt.axis([-5, 5, -3, 3])
plt.xlabel('$y_1(t)$')
plt.ylabel('$y_2(t)$')
plt.legend()
plt.show()

