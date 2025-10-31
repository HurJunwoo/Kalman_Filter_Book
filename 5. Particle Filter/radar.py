import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import norm

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

def fx(x_esti):
    return A @ x_esti

def hx(x_pred):
    z_pred = np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    return np.array([z_pred])

def pf(z_meas, pt, wt):
    pt = fx(pt) + np.random.randn(*pt.shape)

    wt = wt * norm.pdf(z_meas, hx(pt), 10)
    wt = wt / np.sum(wt)

    x_esti = pt @ wt.T

    Npt = pt.shape[1]
    inds = np.random.choice(Npt, Npt, p = wt[0], replace = True)
    pt = pt[:, inds]
    wt = np.ones((1, Npt)) / Npt

    return x_esti, pt, wt

A = np.eye(3) + dt * np.array([[0, 1, 0],
                               [0, 0, 0],
                               [0, 0, 0]])

x_0 = np.array([0, 90, 1100])

Npt = 1000

x_pos_esti_save = np.zeros(n_samples)
x_vel_esti_save = np.zeros(n_samples)
y_pos_esti_save = np.zeros(n_samples)
r_pos_esti_save = np.zeros(n_samples)
r_pos_meas_save = np.zeros(n_samples)

x_pos_pred = 0
z_meas, pt, wt = None, None, None

for i in range(n_samples):
    z_meas, x_pos_pred = get_radar(x_pos_pred)

    if i == 0:
        x_esti = x_0
        pt = x_0.reshape(-1, 1) + 0.1 * x_0.reshape(-1, 1) * np.random.randn(1, Npt)
        wt = np.ones((1, Npt)) / Npt
    else:
        x_esti, pt, wt = pf(z_meas, pt, wt)
    
    x_pos_esti_save[i] = x_esti[0]
    x_vel_esti_save[i] = x_esti[1]
    y_pos_esti_save[i] = x_esti[2]
    r_pos_esti_save[i] = np.sqrt(x_esti[0]**2 + x_esti[2]**2)
    r_pos_meas_save[i] = z_meas

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))

axes[0, 0].plot(time, x_pos_esti_save, 'bo-', label='Estimation (PF)')
axes[0, 0].legend(loc='upper left')
axes[0, 0].set_title('Horizontal Distance: Esti. (PF)')
axes[0, 0].set_xlabel('Time [sec]')
axes[0, 0].set_ylabel('Horizontal Distance [m]')

axes[0, 1].plot(time, y_pos_esti_save, 'bo-', label='Estimation (PF)')
axes[0, 1].legend(loc='upper left')
axes[0, 1].set_title('Vertical Distance: Esti. (PF)')
axes[0, 1].set_xlabel('Time [sec]')
axes[0, 1].set_ylabel('Vertical Distance [m]')

axes[1, 0].plot(time, r_pos_meas_save, 'r*--', label='Measurements', markersize=10)
axes[1, 0].plot(time, r_pos_esti_save, 'bo-', label='Estimation (PF)')
axes[1, 0].legend(loc='upper left')
axes[1, 0].set_title('Radar Distance: Meas. v.s. Esti. (PF)')
axes[1, 0].set_xlabel('Time [sec]')
axes[1, 0].set_ylabel('Radar Distance [m]')

axes[1, 1].plot(time, x_vel_esti_save, 'bo-', label='Estimation (PF)')
axes[1, 1].legend(loc='upper left')
axes[1, 1].set_title('Horizontal Velocity: Esti. (PF)')
axes[1, 1].set_xlabel('Time [sec]')
axes[1, 1].set_ylabel('Horizontal Velocity [m/s]')

plt.savefig('png/radar_pf.png')

    