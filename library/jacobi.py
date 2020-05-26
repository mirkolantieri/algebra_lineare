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
    
    
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        #Jacobi.printSystem(A,b)

        print("Inside Jacobi solver")

        AA = np.asarray(A)
        print("AA shape " + str(AA.shape))
        # AA è una matrice (182435, 3)

        bb = np.asarray(b)
        x = np.asarray(x)
        
        x = np.zeros_like (bb)
        D = np.diag(np.diag(AA))
       
        print("\nnp.diag(AA)")
        print(np.diag(AA))
        # np.diag(AA) restituisce un vettore di 3 elementi [1000.   12.   30.]


        print("\nnp.diag(np.diag(AA))")
        print(np.diag(np.diag(AA)))
        # np.diag(np.diag(AA)) converte il vettore in una matrice diagonale
        #[[1000.    0.    0.]
        #[   0.   12.    0.]
        #[   0.    0.   30.]]

        """
        Qui si sta cercando di sottrarre a una matrice (182435, 3) una matrice (3, 3)
        Il problema è che questo algoritmo funziona su matrici quadrate
        """
        LU = AA - D
        
        start = time.clock() 
        
        for it_count in range(Jacobi.getIteration(k)):
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            x_new = x
            for i in range(AA.shape[0]):
                D_inv = np.diag(1 / np.diag(D))
                x_new = np.dot(D_inv, (bb - np.dot(LU, x)))
                
            if np.allclose(x, x_new, tol):
                break
            x = x_new
        
        end = time.clock() 
    
        print("Computazione in ", end-start)
        print()
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
        error = (np.dot(AA, x) - bb) / bb
        print("Errore rel.:" )
        print(error)

        