import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    """ first the moment gets updated"""    
    mt = np.multiply(beta1, m) + np.multiply((1 - beta1), grad)
    """ then update the second moment """ 
    vt =  np.multiply(beta2, v) + np.multiply((1 - beta2), np.power(grad, 2))

    # update the bias
    mt_cap = mt / (1 - (beta1 ** t))
    vt_cap = vt / (1 - (beta2 ** t))

    # update  the parameter 
    param_new = param - lr * (mt_cap / (np.sqrt(vt_cap) + eps ))  

    return (param_new, mt, vt)
    