import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import io
import math

input_accel_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\11.ARS\ArsAccel.mat')

n_samples = 41500
dt = 0.01
time = np.arange(n_samples) * dt

phi_save = np.zeros(n_samples)
the_save = np.zeros(n_samples)
psi_save = np.zeros(n_samples)

def get_accel(i):
    ax = input_accel_mat['fx'][i][0]
    ay = input_accel_mat['fy'][i][0]
    az = input_accel_mat['fz'][i][0]

    return ax, ay, az

def accel2euler(ax, ay, az, phi, the, psi):
    phi = np.arctan(ay / az)
    the = np.arctan(ax / math.sqrt(ay**2 + az**2))
    psi = psi

    return phi, the, psi

phi, the, psi = 0, 0, 0

for i in range(n_samples):
    ax, ay, az = get_accel(i)
    phi, the, psi = accel2euler(ax, ay, az, phi, the, psi)

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

plt.savefig('png/pose_orientation_accel2.png')