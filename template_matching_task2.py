import os
import cv2 as cv
import numpy as np
from timeit import default_timer
from csv import writer, QUOTE_MINIMAL

templates_directory = "template_images"  # Edit these as needed
images_directory = "images/images"

def build_pyramids(img, pyr_height: int = 3) -> tuple[list, list]:
    temp_image = img.copy()
    temp_gauss = [temp_image]
    for i in range(pyr_height):
        temp_image = cv.pyrDown(temp_image)
        temp_gauss.append(temp_image)

    temp_laplacian = [temp_gauss[-1]]
    for i in range(pyr_height - 1, 0, -1):
        temp_shape = (temp_gauss[i - 1].shape[1], temp_gauss[i - 1].shape[0])
        temp_upscale = cv.pyrUp(temp_gauss[i], dstsize=temp_shape)
        diff = cv.subtract(temp_gauss[i - 1], temp_upscale)
        temp_laplacian.append(diff)

    current = temp_laplacian[0]
    for i in range(1, len(temp_laplacian), 1):
        temp_shape = (temp_laplacian[i + 1].shape[1], temp_laplacian[i + 1].shape[0])
        cv.imshow("Window", current)
        cv.waitKey(0)
        cv.destroyAllWindows()
        current = cv.add(cv.pyrUp(current, dstsize=temp_shape), temp_laplacian)
        cv.imshow("Window", current)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return temp_gauss, temp_laplacian

template_images = []
for filename in os.listdir(templates_directory):  # scale by 1/8
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        im = cv.imread(template_filepath)
        template_images.append(cv.cvtColor(cv.resize(im, (0, 0), fx=1/8, fy=1/8), cv.COLOR_BGR2GRAY))

searchable_images = []  # reads in and grayscales all images
colour_images = []
for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        readImage = cv.imread(image_filepath)
        searchable_images.append(cv.cvtColor(readImage, cv.COLOR_BGR2GRAY))
        colour_images.append(readImage)

timings = []
for imidx, image in enumerate(searchable_images):
    tm = default_timer()
    for template in template_images:
        # Densely matches all template images with main image using the normalised correlation coefficient
        h, w = template.shape
        result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

        threshold = 0.9  # Defines a threshold above which 
        loc = np.where(result >= threshold)

        for x_pt, y_pt in zip(*loc[::-1]):
            cv.rectangle(colour_images[imidx], (x_pt, y_pt), (x_pt + w, y_pt + h), (0, 0, 255), 1)
    tm2 = default_timer()
    timings.append(tm2 - tm)

    cv.imshow('Display image', colour_images[imidx])
    cv.waitKey(0)

with open('times.csv', 'w') as csvfile:
    wt = writer(csvfile, quotechar='|', quoting=QUOTE_MINIMAL)
    wt.writerow(timings)

cv.destroyAllWindows()
