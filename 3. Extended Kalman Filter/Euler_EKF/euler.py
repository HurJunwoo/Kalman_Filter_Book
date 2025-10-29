import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import io

np.random.seed(0)

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
    g = 9.8
    phi = np.arcsin(-ay / (g * np.cos(theta)))
    theta = np.arcsin(ax / g)
    psi = psi

    return phi, theta, psi

H = np.array([[1, 0, 0],
              [0, 1, 0]])

Q = np.array([[0.0001, 0, 0],
              [0, 0.0001, 0],
              [0, 0, 0.1]])

R = np.array([[10, 0],
              [0, 10]])

x_0 = np.array([0, 0, 0])
P_0 = 10*np.eye(3)

dt = 0.01
n_samples = 41500
time = np.arange(0, n_samples * dt, 0.01)

def A_jacob(x_esti):
    A = np.zeros((3, 3))
    phi = x_esti[0]
    theta = x_esti[1]

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    A[0][0] = q*cos_phi*tan_theta - r*sin_phi*tan_theta
    A[0][1] = q*sin_phi / (cos_theta**2) + r*cos_phi / (cos_theta**2)
    A[0][2] = 0

    A[1][0] = -q*sin_phi - r*sin_phi
    A[1][1] = 0
    A[1][2] = 0

    A[2][0] = q*cos_phi / cos_theta - r*sin_phi / cos_theta
    A[2][1] = q*sin_phi*tan_theta/cos_theta + r*cos_phi*tan_theta/cos_theta
    A[2][2] = 0

    A = np.eye(3) + A * dt

    return A

def H_jacob(x_pred):
    return H

def fx(x_esti):
    phi, theta, psi = x_esti

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)
    sec_theta = 1 / np.cos(theta)

    x_dot = np.zeros(3)

    x_dot[0] = p + q * sin_phi * tan_theta
    x_dot[1] = q * cos_phi - r * sin_phi
    x_dot[2] = q * sin_phi * sec_theta + r * cos_phi * sec_theta

    x_pred = x_esti + x_dot * dt

    return x_pred

def hx(x_pred):
    x_esti = H @ x_pred
    return x_esti

def ekf(z_meas, x_esti, P):
    A = A_jacob(x_esti)
    H = H_jacob(x_esti)

    x_pred = fx(x_esti)
    P_pred = A @ P @ A.T + Q

    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    x_esti = x_pred + K @ (z_meas - hx(x_pred))
    
    P = P_pred - K @ H @ P_pred

    return x_esti, P

phi_esti_save = np.zeros(n_samples)
theta_esti_save = np.zeros(n_samples)

for i in range(n_samples):
    p, q, r = get_gyro(i)
    ax, ay, az = get_accel(i)

    phi, theta, psi = accel2euler(ax, ay, az, p, q, r)
    z_meas = np.array([phi, theta])

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = ekf(z_meas, x_esti, P)
    
    phi_esti_save[i] = np.rad2deg(x_esti[0])
    theta_esti_save[i] = np.rad2deg(x_esti[1])


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

axes[0].plot(time, phi_esti_save, 'r', label='Roll ($\\phi$): Estimation (EKF)', markersize=0.2)
axes[0].legend(loc='lower right')
axes[0].set_title('Roll ($\\phi$): Estimation (EKF)')
axes[0].set_xlabel('Time [sec]')
axes[0].set_ylabel('Roll ($\phi$) angle [deg]')

axes[1].plot(time, theta_esti_save, 'b', label='Pitch ($\\theta$): Estimation (EKF)', markersize=0.2)
axes[1].legend(loc='lower right')
axes[1].set_title('Pitch ($\\theta$): Estimation (EKF)')
axes[1].set_xlabel('Time [sec]')
axes[1].set_ylabel('Pitch ($\\theta$) angle [deg]')

plt.savefig('png/pose_orientation_fusion_ekf.png')