""" 
Libreria del metodo lineare di Gauss-Seidel: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time

class GaussSeidel(System):

    def __init__(self):
        return
        
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        #GaussSeidel.printSystem(A,b)
        print("Inside Gauss-Seidel solver")

        A = np.asarray(A)
        b = np.asarray(b)
        x = np.asarray(x)
        
        x = np.zeros_like(b)

        times = np.asarray(x)


        for it_count in range(1, GaussSeidel.getIteration(k)):
            start = time.process_time()
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            x_new = np.zeros_like(x)
            for i in range(A.shape[0]):
                s1 = np.dot(A[i, :i], x_new[:i])
                s2 = np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = np.nan_to_num(b[i] - s1 - s2) / np.nan_to_num(A[i, i])
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        
            end = time.process_time()
            times[it_count:] = (end-start)
        
        GaussSeidel.plotSystem(x, times, "Metodo di Gauss-Seidel")
            
        print()
        print("Soluzione:" )
        print(format(x))
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(format(np.dot(A,x)))
        print()
        error = np.linalg.norm(np.dot(A,x) - b) / np.linalg.norm(b)
        print("Errore rel.:" )
        print(error)
        print()
        print("Computazione in ", times)

        