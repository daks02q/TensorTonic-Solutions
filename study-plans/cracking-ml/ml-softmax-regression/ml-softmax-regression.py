import numpy as np

def softmax_regression(X, y, n_classes, lr=0.01, n_iters=1000):
    """
    Returns: tuple (weights, bias) where weights is a 2D list (d x K) and bias is a list of length K
    """
    X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)
    n, d = X.shape
    K = n_classes
    w = np.zeros((d, K))
    b = np.zeros(K)

    # one hot encode labels
    Y_oh = np.zeros((n,K))
    Y_oh[np.arange(n), y] = 1.0
    
    for i in range(n_iters):
        Z = X @ w + b
        # numerically stablise the softmax 

        Z -= Z.max(axis = 1, keepdims = True)
        exp_Z = np.exp(Z)

        P = exp_Z / exp_Z.sum(axis = 1, keepdims = True)

        #gradients 

        err = P - Y_oh
        dw = ( 1.0 / n) * (X.T @ err)
        db = (1.0 / n ) * err.sum(axis = 0)

        w -= lr * dw
        b -= lr * db

    return (w.tolist(), b.tolist())
