# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt

'''
A = np.array([[2,1,0,1], [1,2,1,0], [0,1,2,1], [1,0,1,2]])
F = np.array([[2],[1],[0],[1]])'''

n = int(input("size of matrix: "))
w = float(input("value of w0: "))

h = 2 * np.pi / n

a = -2 + h**2 * w**2

'''diagonal = np.full(n, a)

# Define the sub-diagonal and superdiagonal elements
sub_diagonal = np.full(n-1, 1)
super_diagonal = np.full(n-1, 1)

# Construct the tridiagonal matrix
A = np.diag(diagonal) + np.diag(sub_diagonal, k=-1) + np.diag(super_diagonal, k=1)

# Set the boundary conditions
A[0, n-1] = 1
A[n-1, 0] = 1'''

#diagonal elements
diagonal = np.array([a for i in range(n)])
A = np.diag(diagonal)
i=1
for i in range(n-1):
    A[i][i+1] = 1
    A[i+1][i] = 1
 
#corner 1s
A[0][n-1] = 1
A[n-1][0] = 1
print('A matrix is:\n')
print(A)


j=0 #check this
F = np.array([3 * np.cos(j*h) for j in range(n)]).reshape(n,1)
print('F matrix is:\n')        
print(F)

x = np.linalg.solve(A, F)
print('U matrix is:\n')
print(x)

#plotting
t = np.linspace(0, 2*np.pi, n)
print (t)
# Plot x vs t
plt.plot(t, x)
plt.xlabel('t')
plt.ylabel('x')
plt.title('Plot of x vs t')
plt.show()