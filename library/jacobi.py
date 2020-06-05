""" 
Libreria del metodo lineare di Jacobi: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time

class Jacobi(System):

    def __init__(self):
        return
    
    
    
    def solver( A, b, x_init, tol, k):
        
        # stampiamo il sistema
        # Jacobi.printSystem(A,b)

        print("Inside Jacobi solver")

        A = np.asarray(A)
        
        bb = np.asarray(b)
        
        
        D = np.diag(np.diag(A))
        LU = A - D
        x = np.asarray(x_init)

        start = time.perf_counter()
        
        for it in range(System.getIteration(k)):
            print("Soluzione iterata {0}:{1}" .format(it, x))
            for i in range(A.shape[0]):
                D_inv = np.diag(1 / np.diag(D))
                x_new = np.dot(D_inv, bb - np.dot(LU, x))
                
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        
        end = time.perf_counter()
        
        Jacobi.plotSystem(x, "Metodo di Jacobi")
            
        print()
        print("Soluzione:")
        print(format(x))
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(format(np.dot(A,x)))
        print()
        error = np.linalg.norm( np.dot(A,x) - bb) / np.linalg.norm(bb)
        print("Errore rel.:")
        print(error)
        print()
        print("Computazione in ", end-start)
