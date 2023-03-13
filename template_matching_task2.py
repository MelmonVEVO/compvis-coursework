import os
import cv2 as cv
import numpy as np

templates_directory = "template_images"  # Edit these as needed
images_directory = "images/images"

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

imidx = 0
for image in searchable_images:
    for template in template_images:  # Edit the following
        w, h = template.shape[::-1]
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)

        threshold = 0.9
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            cv.rectangle(colour_images[imidx], pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    cv.imshow('Display image', colour_images[imidx])
    cv.waitKey(0)

    imidx += 1

cv.destroyAllWindows()
