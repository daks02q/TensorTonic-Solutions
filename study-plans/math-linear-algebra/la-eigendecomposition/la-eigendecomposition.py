import numpy as np

def eigendecompose(A):
    """
    Returns: tuple (eigenvalues, eigenvectors), sorted by descending magnitude.
    """
    A = np.array(A, dtype = float)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvalues = eigenvalues.real 
    eigenvectors = eigenvectors.real
    idx = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # normalize 
    for i in range(eigenvectors.shape[1]): 
        norm = np.linalg.norm(eigenvectors[:, i])
        if norm > 1e-12:
            eigenvectors[:, i]/=norm
    return eigenvalues, eigenvectors