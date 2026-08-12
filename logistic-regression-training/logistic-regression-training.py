import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    #first pass 
    w = np.zeros(X.shape[1])
    b = 0.0
    z = np.dot(X, w) + b
    N = X.shape[0]
    print(type(X))
    print(N)
    p = _sigmoid(z)
    for i in range(steps):     
        error = np.subtract(p, y)
        print(error)
        print(np.transpose(X))
        print(np.dot(np.transpose(X), np.subtract(p, y)))
    
        gradient_w = np.divide(np.dot(np.transpose(X), np.subtract(p, y)), N)
        gradient_y = np.mean(error)

        w = w - lr * gradient_w
        b = b - lr * gradient_y

        z = np.dot(X, w) + b
        p = _sigmoid(z)

    return (w, b)
