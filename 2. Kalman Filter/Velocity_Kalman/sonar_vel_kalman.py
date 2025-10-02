import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import io

np.random.seed(0)

input_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\필터링 기법\data\9.DvKalman\SonarAlt.mat')

n_samples = 500
time_end = 10
dt = time_end / n_samples
A = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]])
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

x_0 = np.array([0, 20])
P_0 = 5 * np.eye(2)


def get_sonar(i):
    z = input_mat['sonarAlt'][0][i]
    return z

def kalman_filter(z_meas, x_esti, P):
    # 1. Prediction
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # 2. Kalman Gain
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # 3. Estimation
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # 4. Error Covariance
    P = (np.eye(len(K)) - K @ H) @ P_pred

    return x_esti, P

time = np.arange(0, time_end, dt)
z_pos_meas_save = np.zeros(n_samples)
x_pos_esti_save = np.zeros(n_samples)
x_vel_esti_save = np.zeros(n_samples)

x_esti, P = None, None

for i in range(n_samples):
    z_meas = get_sonar(i)

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)
    
    z_pos_meas_save[i] = z_meas
    x_pos_esti_save[i] = x_esti[0]
    x_vel_esti_save[i] = x_esti[1]


fig, ax1 = plt.subplots(figsize=(10, 5))
plt.plot(time, z_pos_meas_save, 'r*--', label='Position: Measurements')
plt.plot(time, x_pos_esti_save, 'b-', label='Position: Estimation (KF)')
plt.legend(loc='upper left')
plt.title('Position and Velocity')
plt.xlabel('Time [sec]')
plt.ylabel('Position [m]')

ax2 = ax1.twinx()
plt.plot(time, x_vel_esti_save, 'go-', label='Velocity: Estimation (KF)')
plt.legend(loc='upper right')
plt.ylabel('Velocity [m/s]')
plt.grid(True)
plt.savefig('png/sonar_pos2vel_kf.png')