import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for i in range(n_iters): 
        y_hat = X @ w + b 
        # apply sigmoid function
        Y_hat = 1.0 / ( 1.0 + np.exp(-y_hat))
        #BCE
        error = Y_hat - y
        dw = (1.0 / n) * (X.T @ error)
        db = (1.0 / n) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db

    return w.tolist(), float(b)
