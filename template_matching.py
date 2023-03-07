import os
import cv2 as cv
import numpy as np

templates_directory = "\\template_images\\"  # Edit these as needed
images_directory = "\\images\\"

template_images = []
for filename in os.listdir(templates_directory):
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        template_images.append(cv.imread(template_filepath, 0))

searchable_images = [] # reads in and grayscales all images
for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        searchable_images.append(cv.cvtColor(cv.imread(image_filepath), cv.COLOR_BGR2GRAY))


for image in searchable_images:
    for template in template_images: # Edit the following
        w,h = template.shape[::-1]
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
