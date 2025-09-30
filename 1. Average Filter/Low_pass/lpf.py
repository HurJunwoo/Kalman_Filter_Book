# lpf.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_mat = io.loadmat(r'C:\Users\sgn09\OneDrive\Desktop\AISL\필터링 기법\data\3.LPF\SonarAlt.mat')

def get_sonar(i):
    z = input_mat['sonarAlt'][0][i]
    return z

def low_pass_filter(x_meas, x_esti, alpha):
    x_esti = alpha * x_esti + (1 - alpha) * x_meas
    return x_esti

alpha = 0.7
alpha_2 = 0.2
n_samples = 500
time_end = 10

dt = time_end / n_samples
time = np.arange(0, time_end, dt)
x_meas_save = np.zeros(n_samples)
x_esti_save = np.zeros(n_samples)
x_esti_2_save = np.zeros(n_samples)

x_esti = None
x_esti_2 = None

for i in range(n_samples):
    x_meas = get_sonar(i)
    if i == 0:
        x_esti = x_meas
    else:
        x_esti = low_pass_filter(x_meas, x_esti, alpha)
        x_esti_2 = low_pass_filter(x_meas, x_esti, alpha_2)
    
    x_meas_save[i] = x_meas
    x_esti_save[i] = x_esti
    x_esti_2_save[i] = x_esti_2


plt.plot(time, x_meas_save, 'r*', label='Measured')
plt.plot(time, x_esti_save, 'b-', label='Low-pass Filter (alpha=0.7)')
plt.plot(time, x_esti_2_save, 'g-', label='Low-pass Filter (alpha=0.2)')
plt.legend(loc='upper left')
plt.title('Measured Altitudes v.s. LPF Values')
plt.xlabel('Time [sec]')
plt.ylabel('Altitude [m]')
plt.savefig('png/low_pass_filter.png')