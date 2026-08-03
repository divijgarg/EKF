import numpy as np
import numpy.linalg as la
import math
#this file is to validate my approach to estimating the attitude quaternion q.

#assume q = [q_vec, q_4]

def return_E(q):
    
    E = np.zeros((4,3))
    E[0]= [q[3], -1* q[2], q[1]]
    E[1] = [q[2], q[3], -1 * q[0]]
    E[2] = [-1*q[1], q[0], q[3]]
    E[3] = [-1*q[0], -1*q[1], -1*q[2]]

    return E


def return_cross_matrix(x):

    X = np.zeros((3,3))
    X[0] = [0, -1*x[2], x[1]]
    X[1] = [x[2], 0, -1*x[0]]
    X[2] = [-1*x[1], x[0], 0]

    return X



def return_H_k(q,r):
    A = return_A(q)
    H_k = np.zeros((3,6))
    H_k[0:3, 0:3] = return_cross_matrix(A @ r)

    return H_k



def return_A(q):
    q_vec = q[0:3]
    A = (2*q[3]**2-1)*np.identity(3) - 2*q[3]*return_cross_matrix(q_vec) + 2*np.outer(q_vec, q_vec)
    return A

def return_h_k(A,r):
    return A @r


def return_F(w):
    F = np.zeros((6,6))
    F[0:3, 0:3] = -1*return_cross_matrix(w)
    F[0:3,3:6] = -1*np.identity(3)
    return F

def return_G():
    G = np.zeros((6,6))
    G[0:3, 0:3] = -1*np.identity(3)
    G[3:6,3:6] = -1*np.identity(3)
    return G


def return_Q():
    sigma_v = 0.01
    sigma_u = 0.0001
    Q=np.zeros((6,6))
    Q[0:3,0:3] = sigma_v**2*np.identity(3)
    Q[3:6,3:6] = sigma_u**2*np.identity(3)

    return Q

def main():
    #initialization phase  
    rng = np.random.default_rng(0)
    q_k = np.array([0,0,0,1])
    b_k = np.array([0,0,0])
    P_k = np.diag([0.1]*3 + [1e-4]*3)
    y = np.array([[]])
    w = np.array([[]])
    r1 = np.array([1,0,0])
    r2 = np.array([0,1,0])
    R_k = np.identity(3) * 0.01**2

    
    dt = 0.05
    for i in range(0,50):
        H_k = return_H_k(q_k,r1)
        K_k = gain(P_k,H_k,R_k)
        P_k,q_k, b_k = update(P_k, H_k,K_k,y[i],r1, q_k,b_k)
        q_k, P_k = propogate(b_k,w[i], q_k,P_k,dt)

        H_k = return_H_k(q_k,r2)
        K_k = gain(P_k,H_k,R_k)
        P_k,q_k, b_k = update(P_k, H_k,K_k,y[i],r2, q_k,b_k)
        q_k, P_k = propogate(b_k,w[i], q_k,P_k,dt)
        


def gain(P_k, H_k,R_k):
    term1 = P_k @ H_k.T 
    term2 = H_k @P_k@H_k.T + R_k
    return term1 @ la.inv(term2)

def update(P_k,H_k, K_k,y_k,r_k,q_k,beta_k):
    A_k = return_A(q_k)
    P = (np.identity(6) - K_k @ H_k )@P_k
    delta_x_k = K_k @ (y_k - return_h_k(A_k,r_k))
    delta_theta_k = delta_x_k[0:3]
    delta_beta_k = delta_x_k[3:6]

    q = q_k +0.5*return_E(q_k) @ delta_theta_k
    q = q/la.norm(q)
    beta = beta_k+delta_beta_k

    return P,q,beta


def propogate(beta, w, q,P,dt):
    G = return_G()
    Q= return_Q()

    w_hat = w-beta
    F = return_F(w_hat)
    q_hat_dot = 0.5 * return_E(q) @ w_hat
    P_dot = F @ P + P @ F.T + G @ Q @ G.T

    q_k = q_hat_dot*dt+q
    q_k = q_k/la.norm(q_k)
    P_k = P_dot*dt + P
    return q_k, P_k

main()