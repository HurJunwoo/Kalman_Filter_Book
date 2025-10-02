import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import io
import math

input_gyro_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\11.ARS\ArsGyro.mat')
input_accel_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\11.ARS\ArsAccel.mat')

A = None
H = np.eye(4)
Q = 0.0001 * np.eye(4)
R = 10 * np.eye(4)

x_0 = np.array([1, 0, 0, 0])
x_0 = x_0.T
P_0 = np.eye(4)

n_samples = 41500
dt = 0.01
time = np.arange(n_samples) * dt

phi_esti_save = np.zeros(n_samples)
the_esti_save = np.zeros(n_samples)
psi_esti_save = np.zeros(n_samples)

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

def accel2euler(ax, ay, phi, the, psi):
    g = 9.8
    costhe = np.cos(the)
    phi = np.arcsin(-ay / (g * costhe))
    the = np.arcsin(ax / g)
    psi = psi

    return phi, the, psi

def euler2quaternion(phi, the, psi):
    phi = phi / 2
    the = the / 2
    psi = psi / 2

    cosphi = np.cos(phi)
    costhe = np.cos(the)
    cospsi = np.cos(psi)
    sinphi = np.sin(phi)
    sinthe = np.sin(the)
    sinpsi = np.sin(psi)

    q = np.array([cosphi * costhe * cospsi + sinphi * sinthe * sinpsi,
                  sinphi * costhe * cospsi - cosphi * sinthe * sinpsi,
                  cosphi * sinthe * cospsi + sinphi * costhe * sinpsi,
                  cosphi * costhe * sinpsi - sinphi * sinthe * cospsi])
    
    return q

def quaternion2euler(q):
    # phi_esti = np.arctan2(2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2))
    # the_esti = -np.arcsin(2 * (q[1]*q[3] - q[0]*q[2]))
    # psi_esti = np.arctan2(2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[2]**2 + q[3]**2))

    phi_esti = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    the_esti = -math.pi/2 + 2*np.arctan2(math.sqrt(1 + 2*(q[0]*q[2] - q[1]*q[3])), math.sqrt(1 - 2*(q[0]*q[2] - q[1]*q[3])))
    psi_esti = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

    return phi_esti, the_esti, psi_esti

def kalman_filter(z_meas, x_esti, P):
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    P = P_pred - K @ H @ P_pred

    return x_esti, P

phi, the, psi = 0, 0, 0
x_esti, P = None, None

for i in range(n_samples):
    p, q, r = get_gyro(i)
    A = np.eye(4) + dt / 2 * np.array([[0, -p, -q, r],
                                       [p, 0, r, -q],
                                       [q, -r, 0, p],
                                       [r, q, -p, 0]])
    ax, ay, az = get_accel(i)
    phi, the, psi = accel2euler(ax, ay, phi, the, psi)
    z_meas = euler2quaternion(phi, the, psi)

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)

    phi_esti, the_esti, psi_esti = quaternion2euler(x_esti)

    phi_esti_save[i] = np.rad2deg(phi_esti)
    the_esti_save[i] = np.rad2deg(the_esti)
    psi_esti_save[i] = np.rad2deg(psi_esti)


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

plt.subplot(3, 1, 1)
plt.plot(time, phi_esti_save, 'r', label='Roll ($\\phi$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Roll ($\\phi$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Roll ($\phi$) angle [deg]')

plt.subplot(3, 1, 2)
plt.plot(time, the_esti_save, 'b', label='Pitch ($\\theta$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Pitch ($\\theta$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')

plt.subplot(3, 1, 3)
plt.plot(time, psi_esti_save, 'g', label='Yaw ($\\psi$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Yaw ($\\psi$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw ($\\psi$) angle [deg]')

plt.savefig('png/pose_orientation_fusion_kf.png')