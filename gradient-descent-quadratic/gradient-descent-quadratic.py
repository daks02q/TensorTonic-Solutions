import numpy as np

def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # Write code here
    for step in range(steps):
        gradient = a * 2 * x0 + b 
        x0 = x0 - lr * gradient

    x0 = float(x0)
    return x0