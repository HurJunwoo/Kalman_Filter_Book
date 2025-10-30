import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky

np.random.seed(0)

dt = 0.05
time_end = 20
time = np.arange(0, 20, dt)
n_samples = len(time)

A = np.eye(3) + dt * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])

Q = 0.01 * np.eye(3)
R = np.array([[100]])

x_0 = np.array([[0, 90, 1000]])
P_0 = 100 * np.eye(3)

kappa = 0

def get_radar(x_pos_pred):
    x_vel_true = 100 + np.random.normal(0, 5)
    y_pos_true = 1000 + np.random.normal(0, 10)

    x_pos_pred = x_pos_pred + x_vel_true * dt

    r_pos_v = 0 + x_pos_pred * np.random.normal(0, 0.05)

    r_pos_meas = np.sqrt(x_pos_pred**2 + y_pos_true**2) + r_pos_v

    return r_pos_meas, x_pos_pred

def sigma_point(xm, P, kappa):
    n = xm.shape[0]
    Xi = np.zeros((n, 2*n+1))
    W = np.zeros((2*n+1, 1))

    Xi[:, 0] = xm
    W[0, 0] = kappa / (n + kappa)

    U = cholesky((n + kappa) * P)

    for i in range(n):
        Xi[i+1, :] = xm + U[i, :]
        W[i+1, 0] = 1 / (2*(n + kappa))
    
    for i in range(n):
        Xi[i+n+1, :] = xm - U[i, :]
        W[i+n+1, 0] = 1 / (2*(n + kappa))

    return Xi, W

def UT(Xi, W, noise_cov):
    n = Xi.shape[0]
    k_max = Xi.shape[1]

    x_pred = np.zeros(n)
    P_pred = np.zeros((n, n))
    for i in range(k_max):
        x_pred += W[i, 0] * Xi[i, :]

    for i in range(k_max):
        P_pred += W[i, 0] * (Xi[i, :] - x_pred) @ (Xi[i, :] - x_pred).T
    
    return x_pred, P_pred + noise_cov

def fx(Xi):
    return A @ Xi

def hx(Xi):
    return np.sqrt(Xi[0, :]**2 + Xi[2, :]**2)

def ukf(z_meas, P, x_esti):
    Xi, W = sigma_point(x_esti, P, kappa)

    x_pred, P_pred = UT(fx(Xi), W, Q)

    z_pred, P_z = UT(hx(Xi), W, R)

    n = Xi.shape[0]
    k_max = Xi.shape[1]
    P_xz = np.zeros((n, n))
    fxi = fx(Xi)
    hxi = hx(Xi)

    for i in range(k_max):
        P_xz = W[i, 0] * (fxi - x_pred) @ (hxi - z_pred).T

    K = P_xz @ inv(P_z)

    x_esti = x_pred + K @ (z_meas - z_pred)

    P = P_pred - K @ P_z @ K.T

    return x_esti, P

x_pos_esti_save = np.zeros((1, n_samples))
y_pos_esti_save = np.zeros((1, n_samples))
r_pos_esti_save = np.zeros((1, n_samples))
x_vel_esti_save = np.zeros((1, n_samples))
r_pos_meas_save = np.zeros((1, n_samples))

x_pos_pred = 0
for i in range(n_samples):
    z_meas, x_pos_pred = get_radar(x_pos_pred)

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = ukf(z_meas, P, x_esti)
    
    x_pos_esti_save[0, i] = x_esti[0, 0]
    y_pos_esti_save[0, i] = x_esti[0, 2]
    r_pos_esti_save[0, i] = np.sqrt(x_esti[0, 0]**2 + x_esti[0, 2]**2)
    x_vel_esti_save[0, i] = x_esti[0, 1]
    r_pos_meas_save[0, i] = z_meas


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

axes[0, 0].plot(time, x_pos_esti_save, 'bo-', label='Estimation (UKF)')
axes[0, 0].legend(loc='upper left')
axes[0, 0].set_title('Horizontal Distance: Esti. (UKF)')
axes[0, 0].set_xlabel('Time [sec]')
axes[0, 0].set_ylabel('Horizontal Distance [m]')

axes[0, 1].plot(time, y_pos_esti_save, 'bo-', label='Estimation (UKF)')
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_title('Vertical Distance: Esti. (UKF)')
axes[0, 1].set_xlabel('Time [sec]')
axes[0, 1].set_ylabel('Vertical Distance [m]')

axes[1, 0].plot(time, r_pos_meas_save, 'r*--', label='Measurements', markersize=10)
axes[1, 0].plot(time, r_pos_esti_save, 'bo-', label='Estimation (UKF)')
axes[1, 0].legend(loc='upper left')
axes[1, 0].set_title('Radar Distance: Meas. v.s. Esti. (UKF)')
axes[1, 0].set_xlabel('Time [sec]')
axes[1, 0].set_ylabel('Radar Distance [m]')

axes[1, 1].plot(time, x_vel_esti_save, 'bo-', label='Estimation (UKF)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_title('Horizontal Velocity: Esti. (UKF)')
axes[1, 1].set_xlabel('Time [sec]')
axes[1, 1].set_ylabel('Horizontal Velocity [m/s]')

plt.savefig('png/radar_ukf.png')