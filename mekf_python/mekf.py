import numpy as np
import numpy.linalg as la
import math
import attitude_functions as att

# this file is to validate my approach to estimating the attitude quaternion q.

# assume q = [q_vec, q_4]


def return_H_k(q, r):
    A = att.return_A(q)
    H_k = np.zeros((3, 6))
    H_k[0:3, 0:3] = att.return_cross_matrix(A @ r)

    return H_k


def return_h_k(A, r):
    return A @ r


def return_F(w):
    F = np.zeros((6, 6))
    F[0:3, 0:3] = -1 * att.return_cross_matrix(w)
    F[0:3, 3:6] = -1 * np.identity(3)
    return F


def return_G():
    G = np.zeros((6, 6))
    G[0:3, 0:3] = -1 * np.identity(3)
    G[3:6, 3:6] = -1 * np.identity(3)
    return G


def return_Q():
    sigma_v = 0.01
    sigma_u = 0.0001
    Q = np.zeros((6, 6))
    Q[0:3, 0:3] = sigma_v**2 * np.identity(3)
    Q[3:6, 3:6] = sigma_u**2 * np.identity(3)

    return Q


def read_flight_data(filename):
    with open(filename, "r") as f:
        header = [col.strip() for col in f.readline().strip().split(",")]

    data = np.loadtxt(filename, delimiter=",", comments="#", skiprows=1)
    col = lambda name: header.index(name)

    t = data[:, col("t")]
    y1 = data[:, [col("y1_x"), col("y1_y"), col("y1_z")]]
    y2 = data[:, [col("y2_x"), col("y2_y"), col("y2_z")]]
    w = data[:, [col("w_x"), col("w_y"), col("w_z")]]
    q_true = data[
        :, [col("q_true_1"), col("q_true_2"), col("q_true_3"), col("q_true_4")]
    ]

    return t, y1, y2, w, q_true


def main():
    # initialization phase
    q_k = np.array([-0.0403, 0.0167, 0.3823, 0.9230])
    b_k = np.array([0.0, 0.0, 0.0])
    P_k = np.diag([0.1] * 3 + [1e-4] * 3)

    t, y1, y2, w, q_true = read_flight_data("flight_data.csv")
    # print(t)
    # print(y1)
    # print(w)
    # print(q_true)

    b_true = np.array([0.005, -0.003, 0.002])

    r1 = np.array([0.072718, 0.517414, -0.85264])
    r2 = np.array([0, 0,1])
    R_k = np.identity(3) * 0.01**2

    rows = []

    dt = 0.01
    for i in range(0, len(y1)):
        H_k = return_H_k(q_k, r1)
        K_k = gain(P_k, H_k, R_k)
        if K_k is not None and not np.isnan(K_k).any() and not np.isinf(K_k).any():
            P_k, q_k, b_k = update(P_k, H_k, K_k, y1[i], r1, q_k, b_k)
        
        
        H_k = return_H_k(q_k, r2)
        K_k = gain(P_k, H_k, R_k)

        if K_k is not None and not np.isnan(K_k).any() and not np.isinf(K_k).any():
            P_k, q_k, b_k = update(P_k, H_k, K_k, y2[i], r2, q_k, b_k)
            
        if i % 100 == 0:
            print(i / len(y1)*100, "% complete.")
        q_k, P_k = propogate(b_k, w[i], q_k, P_k, dt)
        rows.append(np.concatenate(([t[i]], q_k, np.diag(P_k), q_true[i], b_k, b_true)))

    header = (
        "t,"
        "q_1,q_2,q_3,q_4,"
        "P_theta_x,P_theta_y,P_theta_z,P_beta_x,P_beta_y,P_beta_z,"
        "q_true_1,q_true_2,q_true_3,q_true_4,"
        "b_x,b_y,b_z,"
        "b_true_x,b_true_y,b_true_z"
    )
    np.savetxt(
        "mekf_log.csv", np.array(rows), delimiter=",", header=header, comments=""
    )


def gain(P_k, H_k, R_k):
    term1 = P_k @ H_k.T
    term2 = H_k @ P_k @ H_k.T + R_k
    try:
        return la.solve(term2, term1.T).T
    except la.LinAlgError:
        return None


def update(P_k, H_k, K_k, y_k, r_k, q_k, beta_k):
    A_k = att.return_A(q_k)
    P = (np.identity(6) - K_k @ H_k) @ P_k
    delta_x_k = K_k @ (y_k - return_h_k(A_k, r_k))
    delta_theta_k = delta_x_k[0:3]
    delta_beta_k = delta_x_k[3:6]

    q = q_k + 0.5 * att.return_E(q_k) @ delta_theta_k
    q = q / la.norm(q)
    beta = beta_k + delta_beta_k

    return P, q, beta


def propogate(beta, w, q, P, dt):
    G = return_G()
    Q = return_Q()

    w_hat = w - beta
    F = return_F(w_hat)
    q_hat_dot = 0.5 * att.return_E(q) @ w_hat
    P_dot = F @ P + P @ F.T + G @ Q @ G.T

    q_k = q_hat_dot * dt + q
    q_k = q_k / la.norm(q_k)
    P_k = P_dot * dt + P
    return q_k, P_k


main()
