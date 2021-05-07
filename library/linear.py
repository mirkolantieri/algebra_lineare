"""
Il file linear.py rappresenta la libreria generale che si va a costruire
come struttura generale, per la risoluzione dei metodi lineari.
Come indicato, servirà come struttura di base, per gli altri metodi, i 
quali verranno definiti nel folder ./library
"""

# importo delle librerie standard essenziali per la lettura delle matrici

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

ITERATION_LIMIT = 50000


class System:

    # il metodo caricherà la matrice in formato array (ovviamente per le matrici sparse verrà fatto l'override del
    # metodo)

    @staticmethod
    def checkIteration(self, k):
        if k <= ITERATION_LIMIT:
            return k
        elif k > ITERATION_LIMIT:
            return ITERATION_LIMIT

    @staticmethod
    def loadMatrix(file):
        b = np.loadtxt(file)
        i = b[:, 0].astype(np.int)
        j = b[:, 1].astype(np.int)
        m = i.max()
        n = j.max()
        return sparse.coo_matrix ((b[:, 2], (i - 1, j - 1)), shape=(m, n)).toarray()

    # il metodo stampa il sistema
    @staticmethod
    def printSystem(A, b):
        mat = np.asanyarray(A)
        bb = np.asarray(b)

        print ("Sistema Lineare:")
        for i in range (mat.shape[0]):
            row = ["{}*x{}".format (mat[i, j], j + 1) for j in range (mat.shape[1])]
            print(" + ".join (row), "=", bb[i])
        print()

    # solver del sistema: ogni sistema iterativo implementerà il metodo in base alle sue specificità
    def solver(self, A, b, x, tol, k):
        return
