# simple_kalman_1.py

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

time_end = 10
dt = 0.2

A = 1
H = 1
Q = 0
R = 4

def get_volt():
    v = np.random.normal(0, 4)
    volt_mean = 14.4
    volt = volt_mean + v
    return volt

def kalman_filter(z_meas, x_esti, P):
    # 1. Prediction
    x_pred = A * x_esti
    P_pred = A * P * A + Q

    # 2. Kalman Gain
    K = P_pred * H / (H * P_pred * H + R)

    # 3. Estimation
    x_esti = x_pred + K * (z_meas - H * x_pred)

    # 4. Error Covariance
    P = P_pred - K * H * P_pred

    return x_esti, P


time = np.arange(0, time_end, dt)
n_samples = len(time)
volt_meas_save = np.zeros(n_samples)
volt_esti_save = np.zeros(n_samples)

x_esti, P = None, None
x_0 = 14
P_0 = 6

for i in range(n_samples):
    z_meas = get_volt()

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)
    
    volt_meas_save[i] = z_meas
    volt_esti_save[i] = x_esti


plt.plot(time, volt_meas_save, 'r*--', label='Measurements')
plt.plot(time, volt_esti_save, 'bo-', label='Kalman Filter')
plt.legend(loc='upper left')
plt.title('Measurements v.s. Estimation (Kalman Filter)')
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.savefig('png/simple_kalman_filter.png')