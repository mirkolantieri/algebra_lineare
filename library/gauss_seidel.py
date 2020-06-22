""" 
Libreria del metodo lineare di Gauss-Seidel: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time


class GaussSeidel (System):

    def solver(self, A, b, x_init, tol, k):

        # stampiamo il sistema

        print ("Inside Gauss-Seidel solver")

        A = np.array(A)
        b = np.asarray(b)
        x = np.asarray(x_init)

        error = None  # inizializzo l'errore a None

        tempo = []

        for it_count in range(1, GaussSeidel.checkIteration(self, k)):

            start = time.process_time ()
            print ("Soluzione iterata {0}:{1}".format (it_count, x))

            # p matrice triangolare inferiore
            p = np.tril (A)

            # residuo scalato
            r = b - np.dot(A, x)

            y = np.dot(np.linalg.inv(p), r)

            x_new = x + y

            x = x_new

            if np.allclose(x, x_new, tol):
                break

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
