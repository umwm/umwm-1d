import numpy as np

def advect(q, c, x):
    """Advects quantity q with advective velocity c in space x."""
    res = np.zeros(q.shape)
    res[1:,:] = np.diff(q, axis=0) / np.diff(x, axis=0)
    res[0,:] = q[0,:] / (x[1,:] - x[0,:])
    return res * c
