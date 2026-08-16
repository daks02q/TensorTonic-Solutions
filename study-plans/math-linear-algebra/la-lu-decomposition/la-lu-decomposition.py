import numpy as np

def lu_decomposition(A):
    """Returns: tuple (L, U) where A = L @ U."""
    A = np.array(A, dtype=float)
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))
    for k in range(n):
        for j in range(k, n):
            U[k, j] = A[k, j] - sum(L[k, m] * U[m, j] for m in range(k))
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - sum(L[i, m] * U[m, k] for m in range(k))) / U[k, k]
    return L, U 
            