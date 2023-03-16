import matplotlib, cv2
import numpy as np
import matplotlib.pyplot as plt


def hough_lines_acc(img, rho_resolution: int = 1, theta_resolution: int = 1):
    """  Accumulates all the sinusoids for image such that hough lines can be formed
    arg img: OpenCV Image
    arg rho_resolution: Minimum resolution for rho parameter (one pixel by default)
    arg theta_resolution: Minimum resolution for theta parameter (one pixel by default)
    returns
        H: Number of sinusoid intersections at each point in the image
        rhos: rho values for each possible location
        thetas: angles for each possible location (computed together with rhos for hough space representation)
    """
    y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes
    height, width = img.shape # we need heigth and width to calculate the diag
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # Gets the length of the leading diagonal of the image
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)  # More efficient than a list comprehension
    # Uses given rho resolution and iterates 
    thetas_deg = np.arange(-90, 90, theta_resolution)
    thetas = np.deg2rad(thetas_deg)
    H = np.zeros((len(rhos), len(thetas)))

    for x, y in zip(x_idxs, y_idxs): # cycle through edge points
        
        for j, theta_curr in enumerate(thetas): # cycle through thetas and calc rho for each one
            rho = (x * np.cos(theta_curr) + y * np.sin(theta_curr)) + img_diagonal
            # rho must be cast to int as it has to map cleanly to image
            H[int(rho), j] += 1

    return H, rhos, thetas


def get_hough_peaks(H, num_peaks: int):
    """
    arg H: array containg number of sinusoid intersections at each point of the image
    arg num_peaks: minimum threshold for number of peaks
    """
    indices = np.argpartition(H.flatten(), -2)[-num_peaks:]
    indices = np.flip(indices, 0)
    return np.vstack(np.unravel_index(indices, H.shape)).T  # REDO THIS IT IS TOO OBSCURE


img = cv2.imread('angle/image1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
min_hysteresis = 100
max_hysteresis = 200
edges = cv2.Canny(gray, min_hysteresis, max_hysteresis)
x1,x2 = edges.shape
count = 0

hough_detection_threshold = 70
lines = cv2.HoughLines(edges, 1, np.pi / 180, hough_detection_threshold)
H, rhos, thetas = hough_lines_acc(edges)
indicies = get_hough_peaks(H, 3)
angles = []
min = 361 # Not 360 as a 360 degree angle is equivalent to a 0 degree angle
max = 0

for ind in indicies:
    # Create lines given our lists of rhos and thetas
    rho = rhos[ind[0]]
    theta = thetas[ind[1]]
    
    # Calculates degrees out of rads
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

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draws line onto the image, no need for capturing return

# Computes correct angle between lines found
for i in range(len(angles)):
    min = angles[i] if angles[i] < min else min
    max = angles[i] if angles[i] > max else max

angle_final = round(360-(max-min) if (max-min) > 180 else max-min)
# More accurate than just casting to int
print(angle_final)  # Our final output

cv2.imshow('img', img)
