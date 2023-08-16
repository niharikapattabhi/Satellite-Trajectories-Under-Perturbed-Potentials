# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:36:47 2023

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

res1 = A2 + B

k=0
F = np.array([(np.cos(k*h)+np.sin(k*h)*np.cos(k*h)) for k in range(3*N)]).reshape((3*N, 1))
print("\n\n this is matrix F: ")
print(F)