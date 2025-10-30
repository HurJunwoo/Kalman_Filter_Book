import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky

np.random.seed(0)

dt = 0.05
time_end = 20
time = np.arange(0, 20, dt)
n_samples = len(time)

def get_radar(x_pos_pred):
    x_vel_true = 100 + np.random.normal(0, 5)
    y_pos_true = 1000 + np.random.normal(0, 10)

    x_pos_pred = x_pos_pred + x_vel_true * dt

    r_pos_v = 0 + x_pos_pred * np.random.normal(0, 0.05)

    r_pos_meas = np.sqrt(x_pos_pred**2 + y_pos_true**2) + r_pos_v

    return r_pos_meas, x_pos_pred

def sigmapoints(xm, P, kappa):
    n = len(xm)
    Xi = np.zeros((n, 2*n+1))
    W = np.zeros(2*n+1)

    Xi[:, 0] = xm

    U = cholesky((n + kappa) * P)

    for i in range(n):
        Xi[:, i+1] = xm + U[:, i]
        Xi[:, i+n+1] = xm - U[:, i]
    
    W[0] = kappa / (n + kappa)

    for i in range(n):
        W[i+1] = W[i+n+1] = 1 / (2 * (n + kappa))

    return Xi, W

def UT(Xi, W, noise):
    xm = np.sum(W * Xi, axis=1)
    # 원래는 W * (Xi - xm.reshape(-1, 1)) @ (Xi - xm.reshape(-1, 1)).T 인데, 행렬 크기 맞춰서 곱하기 위해 순서 및 방법 변경
    P = (Xi - xm.reshape(-1, 1)) @ np.diag(W) @ (Xi - xm.reshape(-1, 1)).T

    return xm, P + noise

def fx(x_esti):
    return A @ x_esti

def hx(x_pred):
    z_pred = np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    return np.array([z_pred])

def ukf(x_esti, P, z_meas):
    Xi, W = sigmapoints(x_esti, P, kappa)

    fxi = fx(Xi)
    hxi = hx(Xi)

    x_pred, P_pred = UT(fxi, W, Q)

    z_pred, P_z = UT(hxi, W, R)
    
    P_xz = W * (fxi - x_pred.reshape(-1, 1)) @ (hxi - z_pred.reshape(-1, 1)).T

    K = P_xz @ inv(P_z)

    x_esti = x_pred + K @ (z_meas - z_pred)

    P = P_pred - K @ P_z @ K.T

    return x_esti, P

time_end = 20
dt = 0.05
time = np.arange(0, 20, dt)
n_samples = len(time)

A = np.eye(3) + dt * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])

Q = 0.01 * np.eye(3)
R = 100

x_0 = np.array([0, 90, 1100])
P_0 = 100 * np.eye(3)

kappa = 0

x_pos_esti_save = np.zeros(n_samples)
y_pos_esti_save = np.zeros(n_samples)
r_pos_esti_save = np.zeros(n_samples)
x_vel_esti_save = np.zeros(n_samples)
r_pos_meas_save = np.zeros(n_samples)

x_pos_pred = 0
for i in range(n_samples):
    z_meas, x_pos_pred = get_radar(x_pos_pred)

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = ukf(x_esti, P, z_meas)
    
    x_pos_esti_save[i] = x_esti[0]
    x_vel_esti_save[i] = x_esti[1]
    y_pos_esti_save[i] = x_esti[2]
    r_pos_esti_save[i] = np.sqrt(x_esti[0]**2 + x_esti[2]**2)
    r_pos_meas_save[i] = z_meas

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