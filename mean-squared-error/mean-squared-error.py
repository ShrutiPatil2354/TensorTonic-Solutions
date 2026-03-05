import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    n = len(y_true)
    mse = 0

    for i in range(n):
        mse += (y_pred[i] - y_true[i]) ** 2

    mse = mse / n
    return mse