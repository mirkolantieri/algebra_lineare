""" 
Libreria del metodo lineare del Gradiente Coniugato: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time

class ConjGrad(System):

    def __init__(self):
        return
        
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        #ConjGrad.printSystem(A,b)
        print("Inside Conjugate Gradient solver")
        
        AA = np.asarray(A)
        bb = np.asarray(b)
        x = np.asarray(x)
        
        
        
        times = np.asarray(x)
        

        
        for it_count in range(1,ConjGrad.getIteration(k)):
            start = time.process_time()
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            for i in range(AA.shape[0]):
                r = bb - np.dot(AA, x)
                d = np.copy(r)
        
                y = np.dot(AA,d)
                z = np.dot(AA,r)
        
                alpha = np.nan_to_num(np.dot(d,r)) / np.nan_to_num(np.dot(d,y))
                x_new = x + np.dot(alpha,d)
                r_new = bb - np.dot(AA,x_new)
                w = np.dot(AA, r_new)
                beta = np.nan_to_num(np.dot(d, w)) / np.nan_to_num(np.dot(d, y))
                d_new = r_new - np.dot(beta,d)
            if np.allclose(x, x_new, tol):
                break
            x = x_new

            end = time.process_time()
            times[it_count:] = end-start
        
        ConjGrad.plotSystem(x, times, "Metodo del Gradiente Coniugato")
            
        print()
        print("Soluzione:" )
        print(x)
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(format(np.dot(AA,x)))
        print()
        error = np.linalg.norm( np.dot(AA,x) - bb) / np.linalg.norm(bb)
        print("Errore rel.:" )
        print(error)

        print()
        print("Computazione in ", times)
        