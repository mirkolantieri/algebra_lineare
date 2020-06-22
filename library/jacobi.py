""" 
Libreria del metodo lineare di Jacobi: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time


class Jacobi(System):

    def solver(self, A, b, x_init, tol, k):

        # stampiamo il sistema
        # Jacobi.printSystem(A,b)

        print("Inside Jacobi solver")

        # transformo i parametri iniziali in array numpy
        A = np.array(A)
        b = np.array(b)
        x = np.array(x_init)

        # inizializzo l'errore a None
        error = None

        tempo = []

        for it in range(1, System.checkIteration(self, k)):
            start = time.process_time()

            print("Soluzione iterata {0}:{1}".format(it, x))

            # creo la diagonale inversa
            diag_inv = np.diag(1 / np.diag(A))

            # residuo scalato
            r = b - np.dot(A, x)

            x_new = x + np.dot(diag_inv, r)

            x = x_new

            # controllo se raggiungo la convergenza
            if np.allclose(x, x_new, tol):
                break

            # np.linalg.norm : metodo che calcola la norma
            error = np.linalg.norm(b - np.dot(A, x)) / np.linalg.norm(b)

            tempo.append(start - time.process_time())

        print()
        print("Soluzione:")
        print(format(x))
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(format(np.dot(A, x)))
        print()
        #print("Errore relativo:\t", "{:.4e}".format(error))
        print ("Errore relativo:\t", error)
        print()
        print("Computazione in ", format(np.abs(sum(tempo))))
