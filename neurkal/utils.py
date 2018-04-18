import numpy as np

def colvec(rowvec):
    v = np.asarray(rowvec)
    return v.reshape(v.size, 1)
