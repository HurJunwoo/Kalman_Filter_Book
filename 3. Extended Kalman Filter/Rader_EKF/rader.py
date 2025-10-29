import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(0)

dt = 0.05
t = np.arange(0, 20, dt)

A = np.eye(3) + dt * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])
Q = np.array([[0, 0, 0],
              [0, 0.001, 0],
              [0, 0, 0.001]])
R = np.array([[10]])

x_0 = np.array([0, 90, 1100])
P_0 = 10 * np.eye(3)

H = np.zeros((1, 3))

nsamples = len(t)

def get_rader(xpos_pred):
    
    vel_true = 100 + np.random.normal(0, 5)
    alt_true = 1000 + np.random.normal(0, 10)

    xpos_pred = xpos_pred + vel_true * dt

    v = 0 + xpos_pred * np.random.normal(0, 0.05)
    rpos_meas = np.sqrt(xpos_pred**2 + alt_true**2) + v

    return rpos_meas, xpos_pred

def H_jacob(x_pred):
    H[0][0] = x_pred[0] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    H[0][1] = 0
    H[0][2] = x_pred[2] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)

    return H

def A_jacob(x_esti):
    return A

def fx(x_esti):
    x_pred = A @ x_esti
    return x_pred

def hx(x_pred):
    z_pred = np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    return np.array([z_pred])

def ekf(z_meas, x_esti, P):
    A = A_jacob(x_esti)
    x_pred = fx(x_esti)
    P_pred = A @ P @ A.T + Q
    H = H_jacob(x_pred)

    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    x_esti = x_pred + K @ (z_meas - hx(x_pred))
    P = P_pred - K @ H @ P_pred

    return x_esti, P

xpos_esti_save = np.zeros(nsamples)
ypos_esti_save = np.zeros(nsamples)
rpos_esti_save = np.zeros(nsamples)
xvel_esti_save = np.zeros(nsamples)
rpos_meas_save = np.zeros(nsamples)

xpos_pred = 0
x_esti, P = None, None
for i in range(nsamples):
    z_meas, xpos_pred = get_rader(xpos_pred)

    if i ==0:
        x_esti, P = x_0, P_0   
    else:
        x_esti, P = ekf(z_meas, x_esti, P)

    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[2]
    rpos_esti_save[i] = np.sqrt(x_esti[0]**2 + x_esti[2]**2)
    xvel_esti_save[i] = x_esti[1]
    rpos_meas_save[i] = z_meas

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

axes[0, 0].plot(t, xpos_esti_save, 'bo-', label='Estimation (EKF)')
axes[0, 0].legend(loc='upper left')
axes[0, 0].set_title('Horizontal Distance: Esti. (EKF)')
axes[0, 0].set_xlabel('Time [sec]')
axes[0, 0].set_ylabel('Horizontal Distance [m]')

axes[0, 1].plot(t, ypos_esti_save, 'bo-', label='Estimation (EKF)')
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_title('Vertical Distance: Esti. (EKF)')
axes[0, 1].set_xlabel('Time [sec]')
axes[0, 1].set_ylabel('Vertical Distance [m]')

axes[1, 0].plot(t, rpos_meas_save, 'r*--', label='Measurements', markersize=10)
axes[1, 0].plot(t, rpos_esti_save, 'bo-', label='Estimation (EKF)')
axes[1, 0].legend(loc='upper left')
axes[1, 0].set_title('Radar Distance: Meas. v.s. Esti. (EKF)')
axes[1, 0].set_xlabel('Time [sec]')
axes[1, 0].set_ylabel('Radar Distance [m]')

axes[1, 1].plot(t, xvel_esti_save, 'bo-', label='Estimation (EKF)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_title('Horizontal Velocity: Esti. (EKF)')
axes[1, 1].set_xlabel('Time [sec]')
axes[1, 1].set_ylabel('Horizontal Velocity [m/s]')

plt.savefig('png/radar_ekf.png')