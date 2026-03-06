import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p=np.array(p)
    y=np.array(y)

    x=(1-p)**gamma*y*np.log(p)
    y=p**gamma*(1-y)*np.log(1-p)

    F=-(x+y)
    return np.mean(F)
    