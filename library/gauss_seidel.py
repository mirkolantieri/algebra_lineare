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
        
        A = np.asarray(A)
        b = np.asarray(b)
        x = np.asarray(x)
        
        x = np.zeros_like (b)
        
        start = time.process_time() 
        
        for it_count in range(1, GaussSeidel.getIteration(k)):
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            x_new = np.zeros_like(x)
            for i in range(A.shape[0]):
                s1 = np.dot(A[i, :i], x_new[:i])
                s2 = np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (b[i] - s1 - s2) / A[i, i]
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        
        end = time.process_time() 
    
        
        GaussSeidel.plotSystem(x, "Metodo di Gauss-Seidel")
            
        print()
        print("Soluzione:" )
        print(format(x))
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(np.dot(A,x))
        print()
        error = (np.dot(A, x) - b) / b
        print("Errore rel.:" )
        print(error)
        print()
        print("Computazione in ", end-start)

        