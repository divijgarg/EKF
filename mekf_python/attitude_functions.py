import numpy as np

def return_E(q):
    return np.array([
        [ q[3], -q[2],  q[1]],
        [ q[2],  q[3], -q[0]],
        [-q[1],  q[0],  q[3]],
        [-q[0], -q[1], -q[2]],
    ])

def return_cross_matrix(x):
    return np.array([
        [    0, -x[2],  x[1]],
        [ x[2],     0, -x[0]],
        [-x[1],  x[0],     0],
    ])

def return_A(q):
    q_vec = q[0:3]
    A = (
        (2 * q[3] ** 2 - 1) * np.identity(3)
        - 2 * q[3] * return_cross_matrix(q_vec)
        + 2 * np.outer(q_vec, q_vec)
    )
    return A
