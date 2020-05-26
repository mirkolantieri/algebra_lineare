"""
Il file linear.py rappresenta la libreria generale che si va a costruire
come struttura generale, per la risoluzione dei metodi lineari.
Come indicato, servirà da struttura di base, per gli altri metodi, i 
quali verranno definiti nel folder /library
"""

# importo delle librerie standard essenziali per la lettura delle matrici

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

ITERATION_LIMIT = 50000

class System:

    def __init__(self):
        return

    # il metodo caricherà la matrice in formato array (ovviamente per le matrici sparse verrà fatto l'override del metodo)
    

    def getIteration(k):
        if k < ITERATION_LIMIT:
            return k
        elif k > ITERATION_LIMIT:
            return ITERATION_LIMIT
    
    def loadMatrix(file):
        B = scipy.io.mmread(file)
        return B.toarray()
    
    # il metodo stampa il sistema    
    def printSystem(A, b):
        mat = np.asanyarray(A)
        bb = np.asarray(b)

        print("Sistema Lineare:")
        for i in range(mat.shape[0]):
            row = ["{}*x{}" .format(mat[i, j], j + 1) for j in range(mat.shape[1])]
            print(" + ".join(row), "=", bb[i])
        print()


    # solver del sistema: ogni sistema iterativo implementerà il metodo in base alle sue specificità
    def solver(A, b, x, tol, k):
        return 
        
    def plotSystem(x, title):
         plt.plot( x, label=title)
         plt.grid()
         plt.legend(loc='best')
         plt.title("Metodi diretti sistemi lineari")