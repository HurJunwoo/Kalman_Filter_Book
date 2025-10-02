import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import inv
from skimage.metrics import structural_similarity

np.random.seed(0)

n_samples = 24
dt = 1

A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Q = 1.0 * np.eye(1)
R = np.array([[50, 0],
              [0, 50]])

x_0 = np.array([0, 0, 0, 0])
P_0 = 100 * np.eye(4)

def get_ball_pose(iimg=0):
    
    imageA = cv2.imread(r'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\10.TrackKalman\Img\bg.jpg')
    imageB = cv2.imread(fr'C:\Users\sgn09\OneDrive\Desktop\AISL\Filtering Methods\data\10.TrackKalman\Img\{iimg + 1}.jpg')

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = max(contours, key=cv2.contourArea)
    M = cv2.moments(cont)

    xc = int(M["m10"] / M["m00"])
    yc = int(M["m01"] / M["m00"])

    v = np.random.normal(0, 15)

    xpos_meas = xc + v
    ypos_meas = yc + v

    return np.array([xpos_meas, ypos_meas])


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


xpos_meas_save = np.zeros(n_samples)
ypos_meas_save = np.zeros(n_samples)
xpos_esti_save = np.zeros(n_samples)
ypos_esti_save = np.zeros(n_samples)

x_esti, P = None, None

for i in range(n_samples):
    z_meas = get_ball_pose(i)

    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = kalman_filter(z_meas, x_esti, P)
    
    xpos_meas_save[i] = z_meas[0]
    ypos_meas_save[i] = z_meas[1]
    xpos_esti_save[i] = x_esti[0]
    ypos_esti_save[i] = x_esti[2]



fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.scatter(xpos_meas_save, ypos_meas_save, s=300, c="r", marker='*', label='Position: Measurements')
plt.scatter(xpos_esti_save, ypos_esti_save, s=120, c="b", marker='o', label='Position: Estimation (KF)')
plt.legend(loc='lower right')
plt.title('Position: Meas. v.s. Esti. (KF)')
plt.xlabel('X-pos. [m]')
plt.ylabel('Y-pos. [m]')
plt.xlim((-10, 350))
plt.ylim((250, -10))
plt.savefig('png/object_tracking_kf.png')