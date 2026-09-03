import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
   
    b = 0.0
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    n,d = X.shape
    w = np.zeros(d)
    for i in range(epochs): 
        y_hat = (X @ w) + b
        loss = y_hat - y
        dw = (2.0/n) * (X.T @ loss)
        db = (2.0/n) * np.sum(loss)
        w = w - (lr * dw)
        b = b - (lr * db)
    return (w, b)
