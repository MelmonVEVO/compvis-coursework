import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    ''' A function for creating a Hough Accumulator for lines in an image. '''
    height, width = img.shape # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64) # empty variable for the rhos and thetas
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)): # cycle through edge points
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # cycle through thetas and calc rho
            rho = int((x * np.cos(thetas[j]) +
                       y * np.sin(thetas[j])) + img_diagonal)
            H[rho, j] += 1

    return H, rhos, thetas


def hough_simple_peaks(H, num_peaks):
    ''' A function that returns the number of indicies = num_peaks of the
        accumulator array H that correspond to local maxima. '''
    indices = np.argpartition(H.flatten(), -2)[-num_peaks:]
    indices = np.flip(indices, 0)
    return np.vstack(np.unravel_index(indices, H.shape)).T


img = cv2.imread('angle/image1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
x1,x2 = edges.shape
count = 0


lines = cv2.HoughLines(edges, 1, np.pi / 180, 70)
H, rhos, thetas = hough_lines_acc(edges)
indicies = hough_simple_peaks(H, 3)
angles = []
min = 361
max = 0

for i in range(len(indicies)):
    # reverse engineer lines from rhos and thetas
    rho = rhos[indicies[i][0]]
    theta = thetas[indicies[i][1]]
    degree = theta * (180/np.pi)
    if rho < 0:
        degree += 180
    if len(angles) == 0:
        angles.append(degree)
    if len(angles) == 1:
        if abs(int(degree) - int(angles[0])) > 2:
            angles.append(degree)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    # these are then scaled so that the lines go off the edges of the image
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


for i in range(0, len(angles)):
    if angles[i] < min:
        min = angles[i]
    if angles[i] > max:
        max = angles[i]

angle = max-min
if angle > 180:
    angle = 360-angle
angleInt = round(angle)
print(int(angleInt))

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


