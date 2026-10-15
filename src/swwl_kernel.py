import numpy as np
from scipy.spatial.distance import cdist
def squared_distances(features):
    d=cdist(features,features,'sqeuclidean'); np.fill_diagonal(d,0); return d
def gaussian_kernel(d,gamma):
    if gamma<=0: raise ValueError('gamma must be positive')
    return np.exp(-gamma*d)
