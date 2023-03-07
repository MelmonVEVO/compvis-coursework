import os
import cv2 as cv

templates_directory = "\\template_images\\"  # Change these filepaths to fit your needs
images_directory = "\\images\\"

template_images = [] # Reads in all template images to this list
for filename in os.listdir(templates_directory):
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        template_images.append(cv.imread(template_filepath, 0))

searchable_images = [] # Reads in all searchable images to this list
for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        searchable_images.append(cv.imread(image_filepath, 0))

for image in searchable_images:
    pass
