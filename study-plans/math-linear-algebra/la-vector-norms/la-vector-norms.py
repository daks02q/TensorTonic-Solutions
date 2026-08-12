import numpy as np

def vector_norms(v):
    """
    Returns: float64 array of shape (3,) containing [L1, L2, L-inf] norms.
    """
    v = np.asarray(v)
    
    a = np.linalg.norm(v, 1)
    b = np.linalg.norm(v)
    c = np.linalg.norm(v,np.inf)
    norms = np.asarray([a,b,c])
    
    return norms