
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
from library.conjgrad import ConjGrad as cg
from library.gradient import Gradient as g

A = [
    [5, 2, 1, 1],
    [2, 6, 2, 1],
    [1, 2, 7, 1],
    [1, 1, 2, 8],
]
b = [29, 31, 26, 19]

x = [1,2,3,4]
tol = 1e-12
k = 1000

x1 = j.solver(A, b, x, tol, k)


x2 = gs.solver(A, b, x, tol, k)


cg = cg.solver(A, b, x, tol, k)

gr = g.solver(A, b, x, tol, k)
