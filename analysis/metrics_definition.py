
import numpy as np
import scipy.sparse

from scipy.spatial.distance import cdist
from scipy.spatial.distance import jaccard
from scipy.spatial.distance import jensenshannon
from scipy.linalg import norm

def simil_fun_cdist(a, b):
    return (cdist(a.reshape(1,len(a)), b.reshape(1,len(b)), 'cosine'))

def simil_fun_euclid(a, b):
    return (np.sum((a - b) * (a - b)))

def simil_fun_hellinger(a, b):
    return (norm(np.sqrt(a) - np.sqrt(b)) / np.sqrt(2))

def simil_fun_jaccard_5(a, b):
    a_bin = a >= 10**(-5)
    b_bin = b >= 10**(-5)
    return jaccard(a, b)

def simil_fun_jaccard_4(a, b):
    a_bin = a >= 10**(-4)
    b_bin = b >= 10**(-4)
    return jaccard(a, b)

def simil_fun_jensen_shannon(a, b):
    return jensenshannon(a, b)

