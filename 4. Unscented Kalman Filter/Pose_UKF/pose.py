import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky
from scipy import io

input_gyro_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\12.EKF\EulerEKF\ArsGyro.mat')
input_accel_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\12.EKF\EulerEKF\ArsAccel.mat')

def get_gyro(i):
    p = input_gyro_mat['wx'][i][0]
    q = input_gyro_mat['wy'][i][0]
    r = input_gyro_mat['wz'][i][0]
    return p, q, r

def get_accel(i):
    ax = input_accel_mat['fx'][i][0]
    ay = input_accel_mat['fy'][i][0]
    az = input_accel_mat['fz'][i][0]
    return ax, ay, az

def accel2euler(ax, ay, az, phi, theta, psi):
    g = 9.81
    phi = np.arcsin(-ay / (g * np.cos(theta)))
    theta = np.arcsin(ax / g)
    psi = psi

    return phi, theta, psi

def sigmapoints(xm, P, kappa):
    n = len(xm)
    Xi = np.zeros((n, 2*n+1))
    W = np.zeros(2*n+1)

    U = cholesky((n + kappa) * P)

    Xi[:, 0] = xm
    W[0] = kappa / (n + kappa)

    for i in range(n):
        Xi[:, i+1] = xm + U[:, i]
        Xi[:, i+n+1] = xm - U[:, i]
    
    for i in range(n):
        W[i+1] = W[i+n+1] = 1 / (2*(n + kappa))
    
    return Xi, W

def UT(Xi, W, noise):
    xm = np.sum(W * Xi, axis = 1)
    P = W * (Xi - xm.reshape(-1, 1)) @ (Xi - xm.reshape(-1, 1)).T

    return xm, P + noise

def fx(x_esti):
    k_max = x_esti.shape[1]

    phi = x_esti[0]
    theta = x_esti[1]
    psi = x_esti[2]

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)
    sec_theta = 1 / np.cos(theta)

    x_dot = np.zeros((3, k_max))
    x_dot[0, :] = p + q * sin_phi * tan_theta + r * cos_phi * tan_theta
    x_dot[1, :] = q * cos_phi - r * sin_phi
    x_dot[2, :] = q * sin_phi * sec_theta + r * cos_phi * sec_theta

    x_pred = x_esti + x_dot * dt

    return x_pred

def hx(x_pred):
    z_pred = H @ x_pred
    return z_pred

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

n_samples = 41500
dt = 0.01
time = np.arange(0, 41500 * dt, dt)

H = np.array([[1, 0, 0],
              [0, 1, 0]])

Q = np.array([[0.0001, 0, 0],
              [0, 0.0001, 0],
              [0, 0, 1]])

R = 10 * np.eye(2)

x_0 = np.zeros(3)
P_0 = np.eye(3)

kappa = 0

phi_esti_save = np.zeros(n_samples)
theta_esti_save = np.zeros(n_samples)
psi_esti_save = np.zeros(n_samples)

phi, theta, psi = 0, 0, 0
x_esti, P = None, None

for i in range(n_samples):
    p, q, r = get_gyro(i)
    ax, ay, az = get_accel(i)
    phi, theta, psi = accel2euler(ax, ay, az, p, q, r)
    z_meas = np.array([phi, theta])

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = ukf(x_esti, P, z_meas)
    
    x_esti = x_esti.flatten()

    phi_esti_save[i] = np.rad2deg(x_esti[0])
    theta_esti_save[i] = np.rad2deg(x_esti[1])
    psi_esti_save[i] = np.rad2deg(x_esti[2])


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

axes[0].plot(time, phi_esti_save, 'r', label='Roll ($\\phi$): Estimation (UKF)', markersize=0.2)
axes[0].legend(loc='lower right')
axes[0].set_title('Roll ($\\phi$): Estimation (UKF)')
axes[0].set_xlabel('Time [sec]')
axes[0].set_ylabel('Roll ($\phi$) angle [deg]')

axes[1].plot(time, theta_esti_save, 'b', label='Pitch ($\\theta$): Estimation (UKF)', markersize=0.2)
axes[1].legend(loc='lower right')
axes[1].set_title('Pitch ($\\theta$): Estimation (UKF)')
axes[1].set_xlabel('Time [sec]')
axes[1].set_ylabel('Pitch ($\\theta$) angle [deg]')

plt.savefig('png/pose_orientation_fusion_ukf.png')