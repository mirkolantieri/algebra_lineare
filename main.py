"""

num_array = list()
num = input("Enter how many elements you want:")
print ('Enter numbers in array: ')
for i in range(int(num)):
    n = input("num :")
    num_array.append(int(n))
print ('ARRAY: ',num_array)

---------------------------------------------------
import numpy as np 
  
R = int(input("Enter the number of rows:")) 
C = int(input("Enter the number of columns:")) 
  
  
print("Enter the entries in a single line (separated by space): ") 
  
# User input of entries in a  
# single line separated by space 
entries = list(map(int, input().split())) 
  
# For printing the matrix 
matrix = np.array(entries).reshape(R, C) 
print(matrix) 

"""

from library.jacobi import Jacobi as j
import numpy as np

A = [
    [5, 2, 1, 1],
    [2, 6, 2, 1],
    [1, 2, 7, 1],
    [1, 1, 2, 8]
]
b = [29, 31, 26, 19]
x = [1,2,3,4]
tol = 1e-12
k = 100

x = j.solver(A, b, x, tol, k)