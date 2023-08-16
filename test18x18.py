# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 21:12:16 2023

@author: Nebula
"""

import numpy as np
import matplotlib.pyplot as plt

N = int(input("size of matrix: "))
w = float(input("value of w0: "))

'''x_values = x[:N, 0]
y_values = x[N:2*N, 0]
z_values = x[2*N:3*N, 0]'''

h = 2 * np.pi / N

#matrix A
diagonal = np.zeros(N)
A = np.diag(diagonal)

for j in range(N):
    for n in range(j+1, N):
        p = ((j*h)-(n*h))/2
        q = 1 / np.tan(p)
        r = j + n
        s = -1 ** r
        d = (s * q)/2
        A[j][n] = d
        A[n][j] = d
    
print("\n\n this is matrix A: ")
#print(A)

# round the matrix to 3 decimal places
A_rounded = np.round(A, decimals=3)
# print the rounded matrix
print(A_rounded)

#matrix A^2
A2 = np.power(A,2)
#print(A2)

#2I
b = 2
diag2 = np.array([b for i in range(N)])
B = np.diag(diag2)
#print(B)

#3I
c = 3
diag3 = np.array([c for i in range(N)])
C = np.diag(diag3)
#print(C)

#5I
d = 5
diag5 = np.array([d for i in range(N)])
D = np.diag(diag5)
#print(D)

#9I
e = 9
diag9 = np.array([e for i in range(N)])
E = np.diag(diag9)
#print(E)

#A^2 + 2I, 3I, 9I
res1 = A2 + B
res2 = A2 + C
res3 = A2+ E

#18x18 zero matrix
P = np.zeros((3*N, 3*N))

#B into A
P[:N, :N] = res1
P[N:2*N, N:2*N] = res2
P[2*N:3*N, 2*N:3*N] = res3

P[:N, N:2*N] = B
P[:N, 2*N:3*N] = C
P[N:2*N, 2*N:3*N] = D
print("\n\n this is matrix P: ")
# round the matrix to 3 decimal places
P_rounded = np.round(P, decimals=3)
# print the rounded matrix
print(P_rounded)

#print(P)

#function f
k=0
F = np.array([np.cos(k*h) for k in range(3*N)]).reshape((3*N, 1))
print("\n\n this is matrix F: ")
print(F)

#solve Pu=f
u = np.zeros((3*N, 1))
u = np.linalg.solve(P, F)
print("\n\n this is matrix u: ")
print(u)

x = u[0:N]
y = u[N:2*N]
z = u[2*N:3*N]

#u[:N] = x
#u[N:2*N] = y
#u[2*N:3*N] = z

print("\n\n this is matrix x: ")
print(np.round(x, decimals=3))
print("\n\n this is matrix y: ")
print(np.round(y, decimals=3))
print("\n\n this is matrix z: ")
print(np.round(z, decimals=3))

#plotting
t = np.linspace(0, 2*np.pi, N)
print("\n\n this is matrix t: ")
print (t)

#plot x vs t
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Plot of x vs t')
plt.show()

#plot x vs t
plt.plot(t, y)
plt.xlabel('t')
plt.ylabel('y')
plt.title('Plot of y vs t')
plt.show()

#plot x vs t
plt.plot(t, z)
plt.xlabel('t')
plt.ylabel('z')
plt.title('Plot of z vs t')
plt.show()
