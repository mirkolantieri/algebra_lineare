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
        
        AA = np.asarray(A)
        bb = np.asarray(b)
        x = np.asarray(x)
        
        r = bb - np.dot(AA, x)
        p = np.copy(r)
        rsold = np.dot(np.transpose(r),r)
        
        start = time.process_time() 
        
        for it_count in range(1,Gradient.getIteration(k)):
            print("Soluzione iterata {0}:{1}" .format(it_count, x))
            for i in range(AA.shape[0]):
                Ap = np.dot(AA,p)
                alpha = rsold / np.exp((np.dot(p, Ap)))  #in caso di valori nan converto in formatp exp
                x = x + (alpha*p)
                r = r - (alpha*Ap)
                rsnew = np.transpose(r) * r
                if np.allclose(r, rsnew, tol):
                    break
                p = r + (rsnew / rsold) * p
                rsold = rsnew

        end = time.process_time() 
    
        
        Gradient.plotSystem(x, "Metodo del Gradiente")
            
        print()
        print("Soluzione:" )
        print(x)
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(np.dot(AA,x))
        print()
        error = (np.dot(AA, x) - bb) / bb
        print("Errore rel.:" )
        print(error)

        print()
        print("Computazione in ", end-start)
        