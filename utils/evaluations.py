import numpy as np


def angular_error(u_true, u_est):
    """
    Angular error between two unit vectors (degrees)
    """
    dot = np.clip(np.dot(u_true, u_est), -1.0, 1.0)
    return np.degrees(np.arccos(dot))