""" 
Libreria del metodo lineare di Gauss-Seidel: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np

class GaussSeidel(System):

    def __init__(self):
        return
        
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        GaussSeidel.printSystem(A,b)
        
        AA = np.asarray(A)
        bb = np.asarray(b)
        x = np.asarray(x)
        
        x = np.zeros_like (bb)
        
        
        for it_count in range(1,GaussSeidel.getIteration(k)):
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            x_new = np.zeros_like(x)
            for i in range(AA.shape[0]):
                s1 = np.dot(AA[i, :i], x_new[:i])
                s2 = np.dot(AA[i, i + 1:], x[i + 1:])
                x_new[i] = (bb[i] - s1 - s2) / AA[i, i]
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        GaussSeidel.plotSystem(x, "Metodo di Gauss-Seidel")
            
        print()
        print("Soluzione:" )
        print(x)
        print()
        print("Valore reale di b:")
        print(bb)
        print()
        print("Valore computato di b:")
        print(np.dot(AA,x))
        print()
        error = np.dot(AA, x) - bb
        print("Errore rel.:" )
        print(error)

        