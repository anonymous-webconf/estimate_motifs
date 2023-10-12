import numpy as np

def get_stationary_distribution (matrix):
    matrix = np.array(matrix)
    for row in matrix:
        assert sum(row) >= 0.999 and sum(row) <=1.0005
    #We have to transpose so that Markov transitions correspond to right multiplying by a column vector.  np.linalg.eig finds right eigenvectors.
    evals, evecs = np.linalg.eig(matrix.T)
    evec1 = evecs[:,np.isclose(evals, 1)]

    #Since np.isclose will return an array, we've indexed with an array
    #so we still have our 2nd axis.  Get rid of it, since it's only size 1.
    evec1 = evec1[:,0]

    stationary = evec1 / evec1.sum()

    #eigs finds complex eigenvalues and eigenvectors, so you'll want the real part.
    stationary = stationary.real
    assert sum(stationary) >= 0.99 and sum (stationary) <= 1.005
    return stationary.real