""" 
Libreria del metodo lineare del Gradiente Coniugato: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time

class ConjGrad(System):

    def solver(self, A, b, x_init, tol, k):

        # stampiamo il sistema
        print("Inside Conjugate Gradient solver")

        A = np.array (A)
        b = np.array (b)
        x = np.array (x_init)

        error = None  # inizializzo l'errore a None

        tempo = []

        for it_count in range(1, ConjGrad.checkIteration(self, k)):
            start = time.process_time ()
            print ("Soluzione iterata {0}:{1}".format (it_count, x))

            # residuo scalato
            r = b - np.dot(A, x)
            d = np.copy(r)
            y = np.dot(A, d)
            z = np.dot(A, r)

            alpha = np.exp(np.dot(d, r)) / np.exp(np.dot(d, y))
            x_new = x + np.dot(alpha, d)

            r_new = b - np.dot(A, x_new)

            w = np.dot(A, r_new)
            beta = np.exp(np.dot(d, w)) / np.exp(np.dot(z, d))

            d = r_new - np.dot(beta, d)

            x = x_new

            if np.allclose(x, x_new, tol):
                break

            error = np.linalg.norm(b - np.dot (A, x)) / np.linalg.norm(b)

            tempo.append (start - time.process_time ())

        print()
        print("Soluzione:")
        print(format (x))
        print()
        print("Valore reale di b:")
        print(b)
        print()
        print("Valore computato di b:")
        print(format (np.dot (A, x)))
        print()

        # print("Errore relativo:\t", "{:.4e}".format(error))
        print ("Errore relativo:\t", error)
        print()
        print("Computazione in ", format(np.abs(sum(tempo))))
