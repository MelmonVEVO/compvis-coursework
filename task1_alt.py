import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

IMAGE_PATH = "C:\\Users\\Gabriel\\Downloads\\Task1Dataset\\angle\\image7.png"

img = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
edges = cv.Canny(img, 50, 200, None, 3)
coloured_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

def get_angle(point_1, point_2, point_3, point_4):
    y_change = (point_1[1] - point_2[1])
    x_change = (point_1[0] - point_2[0])
    if x_change == 0:
        grad_1 = math.inf
    else:
        grad_1 = y_change / x_change

    y_change = (point_3[1] - point_4[1])
    x_change = (point_3[0] - point_4[0])
    if x_change == 0:
        grad_2 = math.inf
    else:
        grad_2 = y_change / x_change

    angle = math.atan((grad_1 - grad_2) / (1 + grad_1 * grad_2))
    return math.degrees(angle)

num_lines = 0
pixel_threshhold = 130
lines = []
while num_lines < 2:    
    lines = cv.HoughLines(edges, 1, np.pi / 180, pixel_threshhold, None, 0, 0)
    print(num_lines)
    if lines is None:
        num_lines = 0
    else:
        num_lines = len(lines)
    pixel_threshhold -= 5


start_points = []
end_points = []
for i in range(len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    alpha_const = math.cos(theta)
    beta_const = math.sin(theta)
    x_pos = alpha_const * rho
    y_pos = beta_const * rho
    start_points.append((int(x_pos + 1000 * (-beta_const)), int(y_pos + 1000 * alpha_const)))
    end_points.append((int(x_pos - 1000 * (-beta_const)), int(y_pos - 1000 * alpha_const)))
    
    cv.line(coloured_edges, start_points[-1], end_points[-1], (0,0,255), 3, cv.LINE_AA)

print(start_points, end_points)

angles = []
for pair_start in range(len(start_points) - 1):
    angles.append(get_angle(start_points[pair_start], end_points[pair_start], start_points[pair_start + 1], end_points[pair_start + 1]))
    
print(angles)
print(max(angles))
cv.imshow("FINAL", coloured_edges)
