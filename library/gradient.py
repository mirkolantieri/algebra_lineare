""" 
Libreria del metodo lineare del Gradiente: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time

class Gradient(System):

    def __init__(self):
        return
        
    
    def solver( A, b, x, tol, k):
        
        # stampiamo il sistema
        #Gradient.printSystem(A,b)
        print("Inside Gradient solver")
        
        AA = np.asarray(A)
        bb = np.asarray(b)
        x = np.asarray(x)

        times = np.asarray(x)
        

        for it_count in range(1,Gradient.getIteration(k)):
            start = time.process_time()
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            for i in range(AA.shape[0]):
                r = bb - np.dot(AA, x)
                y = np.dot(AA, r)
                alpha = np.nan_to_num(np.matmul(np.transpose(r), r)) / np.nan_to_num(np.matmul(np.transpose(r), y))

                x_new = x + np.dot(alpha,r)
            if np.allclose(x, x_new, tol):
                break
            x = x_new

            end = time.process_time()

            times[it_count:] = end-start
        
        Gradient.plotSystem(x, times, "Metodo del Gradiente")
            
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
        