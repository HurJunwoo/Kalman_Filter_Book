import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\11.ARS\ArsGyro.mat')

n_samples = 41500
dt = 0.01

time = np.arange(n_samples) * dt
phi_save = np.zeros(n_samples)
the_save = np.zeros(n_samples)
psi_save = np.zeros(n_samples)

phi, the, psi = 0, 0, 0

def get_gyro(i):
    p = input_mat['wx'][i][0]
    q = input_mat['wy'][i][0]
    r = input_mat['wz'][i][0]

    return p, q, r

def euler_gyro(phi, the, psi, p, q, r, dt):
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_the = np.tan(the)
    cos_the = np.cos(the)

    phi = phi + dt * (p + sin_phi * tan_the * q + cos_phi * tan_the * r)
    the = the + dt * (cos_phi * q - sin_phi * r)
    psi = psi + dt * (sin_phi / cos_the * q + cos_phi / cos_the * r)

    return phi, the, psi

for i in range(n_samples):
    p, q, r = get_gyro(i)
    phi, the, psi = euler_gyro(phi, the, psi, p, q, r, dt)
    phi_save[i] = np.rad2deg(phi)
    the_save[i] = np.rad2deg(the)
    psi_save[i] = np.rad2deg(psi)

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

plt.subplot(3, 1, 1)
plt.plot(time, phi_save, 'r', label='Roll ($\\phi$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Roll ($\\phi$)')
plt.xlabel('Time [sec]')
plt.ylabel('Roll ($\phi$) angle [deg]')

plt.subplot(3, 1, 2)
plt.plot(time, the_save, 'b', label='Pitch ($\\theta$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Pitch ($\\theta$)')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')

plt.subplot(3, 1, 3)
plt.plot(time, psi_save, 'g', label='Yaw ($\\psi$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Yaw ($\\psi$)')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw ($\\psi$) angle [deg]')

plt.savefig('png/pose_orientation_gyro.png')
