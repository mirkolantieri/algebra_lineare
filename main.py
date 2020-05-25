""" File main.py
"""


from library.jacobi import Jacobi as j
from library.gauss_seidel import GaussSeidel as gs
from library.conjgrad import ConjGrad as cg
from library.gradient import Gradient as g
import scipy.io
import numpy as np


A =  np.array([[10., -1., 2., 0.],
 [-1., 11., -1., 3.],
 [2., -1., 10., -1.],
 [0.0, 3., -1., 8.]])

b = np.asarray([1, 2, -1, 1])

x = np.dot(A,b)
tol = 1e-12
k = 1000



x1 = j.solver(A, b, x, tol, k)



"""

x2 = gs.solver(A, b, x, tol, k)

gr = g.solver(A, b, x, tol, k)

cg = cg.solver(A, b, x, tol, k)

"""