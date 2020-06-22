""" 
Libreria del metodo lineare del Gradiente: come si nota
alcuni dei metodi sono già visti nella classe astratta "Linear.py",
quindi possiamo fare tranquillamente l'override dei metodi
"""

from library.linear import System
import numpy as np
import time


class Gradient (System):

    def solver(self, A, b, x_init, tol, k):

        # stampiamo il sistema

        print("Inside Gradient solver")

        A = np.array(A)
        b = np.array(b)
        x = np.array(x_init)

        error = None  # inizializzo l'errore a None

        tempo = []

        for it_count in range(1, Gradient.checkIteration(self, k)):
            start = time.process_time()
            print("Soluzione iterata {0}:{1}".format(it_count, x))

            # residuo scalato
            r = b - np.dot(A, x)
            r_t = np.transpose(r)
            y = np.dot(A, r)

            c1 = np.dot(r, r_t)
            c2 = np.dot(r_t, y)

            alpha = np.exp(c1) / np.exp(c2)

            x_new = x + np.dot(alpha, r)

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
