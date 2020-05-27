""" File main.py
"""


from library.jacobi import Jacobi as j
from library.linear import System as s
from library.gauss_seidel import GaussSeidel as gs
from library.conjgrad import ConjGrad as cg
from library.gradient import Gradient as g
import numpy as np


A = np.array([[10., -1., 2., 0.],
              [-1., 11., -1., 3.],
              [2., -1., 10., -1.],
              [0., 3., -1., 8.]])

b = np.array([6., 25., -11., 15.])

x = np.dot(A,b)
tol = 1e-4
k = 10000


#j.solver(A, b, x, tol, k)

#gs.solver(A, b, x, tol, k)

#g.solver(A, b, x, tol, k)

#cg.solver(A, b, x, tol, k)

###############################################################################

B = s.loadMatrix('data/spa1.mtx')

xx = np.ones(len(B))

bb = np.dot(B, xx)

j.solver(B, bb, xx, tol, k)

#gs.solver(B, bb, xx, tol, k)


#g.solver(B, bb, xx, tol, k)

#cg.solver(B, bb, xx, tol, k)

