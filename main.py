""" File main.py
"""

from library.jacobi import Jacobi
from library.linear import System as s
from library.gauss_seidel import GaussSeidel
from library.conjgrad import ConjGrad
from library.gradient import Gradient
import numpy as np

# parte semplice di analisi: matrice simmetrica di dimensioni 4x4

j = Jacobi ()
gs = GaussSeidel ()
g = Gradient ()
cg = ConjGrad ()

k = 30000
tol = 1e-10
"""
A = np.array([[10., -1., 2., 0.],
               [-1., 11., -1., 3.],
               [2., -1., 10., -1.],
               [0., 3., -1., 8.]])

b = np.array ([6., 25., -11., 15.])

x = np.zeros(len(b))

s.printSystem(A, b)

# solutori (metodi iterativi eseguiti uno dopo l'altro)

sol1 = j.solver(A, b, x, tol, k)
sol2 = gs.solver(A, b, x, tol, k)
sol3 = g.solver(A, b, x, tol, k)
sol4 = cg.solver(A, b, x, tol, k)


# parte dei file .mtx : applichiamo i solutori


"""
B = s.loadMatrix('data/vem2.mtx')

x1 = np.ones(len(B))

x0 = np.zeros(len(B))

bb = np.matmul(B, x1)


g.solver(B, bb, x0, tol, k)