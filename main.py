import numpy as np 
"""
  
R = int(input("Inserire il numero delle righe:")) 
C = int(input("Inserire il numero delle colonne:")) 
  
  
print("Inserire gli elementi in una singola linea (separare con dello spazio): ") 
  
# User input of entries in a  
# single line separated by space 
entries = list(map(int, input().split())) 
  
# For printing the matrix 
matrix = np.array(entries).reshape(R, C) 
print(matrix) 


"""


from library.jacobi import Jacobi as j
from library.gauss_seidel import GaussSeidel as gs

A = [
    [5, 2, 1, 1],
    [2, 6, 2, 1],
    [1, 2, 7, 1],
    [1, 1, 2, 8],
]
b = [29, 31, 26, 19]

x = [1,2,3,4]
tol = 1e-9
k = 1000

x1 = j.solver(A, b, x, tol, k)

A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])
# initialize the RHS vector
b = np.array([6., 25., -11., 15.])


x2 = gs.solver(A, b, x, tol, k)
