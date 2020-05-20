from library.linear import System
import numpy as np



class Jacobi(System):

    def solver(self, A, b, x, tol, k):
        mat = np.asarray(A)
        bb = np.asarray(b)

        Jacobi.printSystem(A,b)

        x = np.zeros_like(bb)
        for it in range(Jacobi.getIteration(k)):
            print("Current solution:" , x)
            x_new = np.zeros_like (x)

            for i in range(mat.shape[0]):
                s1 = np.dot(mat[i, :i], x[:i])
                s2 = np.dot(mat[i, i + 1:], x[i + 1:])
                x_new[i] = (bb[i] - s1 - s2) / mat[i, i]
            if np.allclose (x, x_new, tol, rtol=0.):
                break
            x = x_new
        print("Solution:\n " + x )
        error = np.dot(A, x) - b
        print("Error:" )
        print(error)
