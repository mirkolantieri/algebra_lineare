""" 
Libreria del metodo lineare di Jacobi: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np

class Jacobi(System):

    def __init__(self):
        return
    
    
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        Jacobi.printSystem(A,b)
        
        AA = np.asarray(A)
        bb = np.asarray(b)
        x = np.asarray(x)
        
        x = np.zeros_like (bb)
        D = np.diag(np.diag(AA))
        LU = A - D
        
        
        for it_count in range(Jacobi.getIteration(k)):
            print("Soluzione iterata:" , x)
            x_new = x
            for i in range(AA.shape[0]):
                D_inv = np.diag(1 / np.diag(D))
                x_new = np.dot(D_inv, (bb - np.dot(LU, x)))
                
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        Jacobi.plotSystem(x, "Metodo di Jacobi")
            
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

        