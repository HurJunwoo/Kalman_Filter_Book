# avgfilter.py

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def get_volt():
    v = np.random.normal(0, 4)
    volt_mean = 14.4
    volt = volt_mean + v
    return volt

def average_filter(k, volt, volt_mean):
    alpha = (k - 1) / k
    volt_mean = alpha * volt_mean + (1 - alpha) * volt
    return volt_mean

time_end = 10
dt = 0.2

time = np.arange(0, time_end, dt)
n_samples = len(time)

x_mean_save = np.zeros(n_samples)
x_avg_save = np.zeros(n_samples)

volt_avg = 0

for i in range(n_samples):
    k = i + 1
    volt = get_volt()
    volt_avg = average_filter(k, volt, volt_avg)
    x_mean_save[i] = volt
    x_avg_save[i] = volt_avg

    print(x_mean_save[i])
    print(x_avg_save[i])

plt.plot(time, x_mean_save, 'r*', label='Measured')
plt.plot(time, x_avg_save, 'b-', label='Average')
plt.legend(loc='upper left')
plt.title('Measured Voltages v.s. Average Filter Values')
plt.xlabel('Time [sec]')
plt.ylabel('Volt [V]')
plt.savefig('png/average_filter.png')