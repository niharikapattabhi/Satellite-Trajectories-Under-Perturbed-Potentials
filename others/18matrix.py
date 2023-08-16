# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:18:00 2023

@author: Nebula
"""

import numpy as np
import matplotlib.pyplot as plt

N = int(input("size of matrix: "))
w = float(input("value of w0: "))

h = 2 * np.pi / N

#matrix A
diagonal = np.zeros(N)
A = np.diag(diagonal)

j = 0 
n = 0
for j in range(N-1):
    for n in range(N-1):
        if j!=n:
            p = ((j*h)-(n*h))/2
            q = 1 / np.tan(p)
            r = j + n
            s = -1 ** r
            d = (s * q)/2
            A[j][n] = d
    
print(A)

#matrix A^2
A2 = np.power(A,2)
print(A2)

#2I
b = 2
diag2 = np.array([b for i in range(N)])
B = np.diag(diag2)
print(B)

#3I
c = 3
diag3 = np.array([c for i in range(N)])
C = np.diag(diag3)
print(C)

#5I
d = 5
diag5 = np.array([d for i in range(N)])
D = np.diag(diag5)
print(D)

#9I
e = 9
diag9 = np.array([e for i in range(N)])
E = np.diag(diag9)
print(E)

#A^2 + 2I, 3I, 9I
res1 = A + B
res2 = A + C
res3 = A + E

#18x18 zero matrix
P = np.zeros((18, 18))

#B into A
P[:6, :6] = res1
P[6:12, 6:12] = res2
P[12:18, 12:18] = res3

P[:6, 6:12] = B
P[:6, 12:18] = C
P[6:12, 12:18] = E


print(P)

#function f
k=0
F = np.array([np.cos(k*h) for k in range(3*N)]).reshape(3*N,1)
print(F)

#solve Au=f
x = np.linalg.solve(P, F)
#u
print(x)

#plotting
t = np.linspace(0, 2*np.pi, 3*N)
print (t)

#plot x vs t
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Plot of x vs t')
plt.show()