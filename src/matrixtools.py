import pprint

import numpy
import numpy as np


# el proceso de Gram-Schmidt se utiliza para ortonormalizar
# las columnas de una matriz
# https://es.wikipedia.org/wiki/Factorizaci%C3%B3n_QR#Mediante_el_m%C3%A9todo_de_ortogonalizaci%C3%B3n_de_Gram-Schmidt
def gramschmidt(matrix):
    m, n = matrix.shape
    # Initialize matrix with zeros
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    A = np.copy(matrix)
    for i in range(n):
        v = A[:, i]
        norm = np.linalg.norm(v)
        Q[:, i] = v / norm
        R[i, i] = norm
        for j in range(i+1,n):
            q = numpy.transpose(Q[:, i])
            R[i, j] = q.dot(A[:, j])
            A[:, j] = A[:, j] - R[i, j] * q
    return Q, R

# Same example as wikipedia
matrix = numpy.array([[12, -51, 4], [6, 167, -68], [-4, 24, -41]])
pprint.pprint(matrix)
[q,r] = gramschmidt(matrix)
pprint.pprint(q)
pprint.pprint(r)