import numpy as np
import numpy.linalg as la
import attitude_functions as att


rng = np.random.default_rng(23)

dt = 0.05           
N = 4000             # number of samples
sigma_v = 0.01       # gyro white-noise density
meas_noise = 0.01    # vector-measurement noise std

b_true = np.array([0.005, -0.003, 0.002])   # (rad/s)
r1 = np.array([1.0, 0.0, 0.0])              # ref 1
r2 = np.array([0.0, 1.0, 0.0])              # ref 2

def w_true_at(t):
    # return np.array([0.02*np.sin(0.05*t),
    #                  -0.01*np.cos(0.03*t),
    #                  0.03*np.sin(0.02*t + 1.0)])
    return np.array([0.2,0,0])

q_true = np.array([0.0, 0.0, 0.0, 1.0])
rows = []
for i in range(N):
    t = i*dt

    A_t = att.return_A(q_true)
    y1 = A_t @ r1 + rng.normal(0, meas_noise, 3)
    y2 = A_t @ r2 + rng.normal(0, meas_noise, 3)
    w_meas = w_true_at(t) + b_true + rng.normal(0, sigma_v/np.sqrt(dt), 3)

    rows.append(np.concatenate(([t], y1, y2, w_meas, q_true)))

    sub = dt/10
    for k in range(10):
        w = w_true_at(t + k*sub)
        q_true = q_true + 0.5*att.return_E(q_true) @ w * sub
        q_true = q_true/la.norm(q_true)

header = ("t, y1_x, y1_y, y1_z, y2_x, y2_y, y2_z, w_x, w_y, w_z, "
          "q_true_1, q_true_2, q_true_3, q_true_4   "
          "| dt=0.05  r1=[1,0,0]  r2=[0,1,0]  true bias=[0.005,-0.003,0.002]  "
          "meas_noise=0.01  sigma_v=0.01")

np.savetxt("sample_attitude_data.csv", np.array(rows), delimiter=",",
           header=header, fmt="%.8f")

print("done writing sample_attitude_data.csv")