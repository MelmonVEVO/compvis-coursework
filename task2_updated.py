import os
import cv2 as cv
import numpy as np

templates_directory = "Task2Dataset/Training/png"  # Edit these as needed
images_directory = "Task2Dataset/TestWithoutRotations/images"

p_size = 2


def get_jaccard(coords: np.ndarray, thresh: float) -> list:
    """ Non-maximum suppression algorithm using the jaccard overlap index to eliminate multiple detections
    :param coords: list of all points and their confidence scores
    :param thresh: jaccard threshold
    :return: list of all scores above threshold
    """
    if coords.shape == (0,):
        return []

    # Gets numpy arrays of each individual coordinate. x1y1 are the top left corner and x2y2 are bot right
    x1 = coords[:, 0]
    y1 = coords[:, 1]
    x2 = coords[:, 2]
    y2 = coords[:, 3]
    # Gets correlation coefficient of each box to ensure only the highest are kept
    corr_coeffs = coords[:, 4]

    # Parallel processes the areas of all boxes. These are always all identical for task 2 but not necessarily for 3.
    area_arr = (x2 - x1) * (y2 - y1)

    # Gets indices that each element from the correlation coefficients would represent when sorted for use later
    ordered_coeffs = np.argsort(corr_coeffs)

    final = []
    # Performs huge parallell processing to speed up non maximum suppression
    while len(ordered_coeffs) >= 1:
        # Gets the index of the highest correlation value, this will always be a match the first time around
        highest_ind = ordered_coeffs[-1]
        final.append(coords[highest_ind][:-1])  # Removes score while adding coords to final

        # Deletes the value we have added to the page from our bounding boxes to search through
        ordered_coeffs = np.delete(ordered_coeffs, -1)

        # If removing this has reduced the array size to zero there is no need for comparison and the values can return
        if len(ordered_coeffs) == 0:
            return final

        # Gets the areas in sorted order of correlation coefficient.
        # This operation is useless for uniform sizes but for multiple sizes is vital.
        box_orders = area_arr[ordered_coeffs]

        # extracts the coordinates in order, this could be done on the whole original array but wastes the sort on score
        x1_ord = x1[ordered_coeffs]
        y1_ord = y1[ordered_coeffs]
        x2_ord = x2[ordered_coeffs]
        y2_ord = y2[ordered_coeffs]

        # Finds the intersection points
        x1_high, y1_high = np.maximum(x1_ord, x1[highest_ind]), np.maximum(y1_ord, y1[highest_ind])
        x2_high, y2_high = np.minimum(x2_ord, x2[highest_ind]), np.minimum(y2_ord, y2[highest_ind])

        # Parallel calculates the widths and heights of each intersection, with non intersecting set to 0
        widths = np.maximum(x2_high - x1_high, 0)
        heights = np.maximum(y2_high - y1_high, 0)

        # Finds intersection areas
        inter_area = widths * heights

        # Finds union of areas of boxes. Will be lower for objects with an intersection.
        union_area = (box_orders - inter_area) + area_arr[highest_ind]

        # Calculates final jaccard score
        jaccard_scores = inter_area / union_area

        # Any records with scores too high (i.e. too much overlap) are discarded
        ordered_coeffs = ordered_coeffs[jaccard_scores < thresh]

    return final


def build_pyramids(img: np.ndarray, pyr_height: int = 3) -> tuple[list, list]:
    temp_image = img.copy()
    temp_gauss = [temp_image]
    # Forms the gaussian pyramid, starting with the original image in the pyramid
    for i in range(pyr_height):
        # The pyrDown function uses a gaussian kernel on the image then downsamples to half its size by default
        temp_image = cv.pyrDown(temp_image)
        # Adds downsampled image to gauss pyramid
        temp_gauss.append(temp_image)

    # Uses gaussian pyramid to construct a laplacian pyramid, this is a little inefficient but not horribly so
    temp_laplacian = [temp_gauss[-1]]
    for i in range(pyr_height - 1, 0, -1):
        # Shape is backwards from numpy to opencv so they have to be done backwards
        temp_shape = (temp_gauss[i - 1].shape[1], temp_gauss[i - 1].shape[0])
        # The inverse operation of pyrDown is performed via pyrUp.
        # The absolute size to upscale to is specified here.
        # This is because a naive doubling of size
        # does not account for images with odd numbers of pixels before their downsampling
        temp_upscale = cv.pyrUp(temp_gauss[i], dstsize=temp_shape)
        # Subtract the upscaled image from the one in our gaussian pyramid. This gets their absolute difference.
        diff = cv.subtract(temp_gauss[i - 1], temp_upscale)
        # Adds this absolute difference to the laplacian pyramid.
        temp_laplacian.append(diff)

    return temp_laplacian


template_images = []
lap_templates = []
for filename in os.listdir(templates_directory):  # scale by 1/8
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        im = cv.imread(template_filepath)
        template_images.append(cv.cvtColor(cv.resize(im, (0, 0), fx=1 / 8, fy=1 / 8), cv.COLOR_BGR2GRAY))
        lap_templates.append(build_pyramids(cv.resize(im, (0, 0), fx=1 / 8, fy=1 / 8), p_size))

searchable_images = []  # reads in and grayscales all images
colour_images = []
lap_pyramids = []  # List of laplacian pyramids

for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        readImage = cv.imread(image_filepath)
        searchable_images.append(cv.cvtColor(readImage, cv.COLOR_BGR2GRAY))
        colour_images.append(readImage)
        # Adds the laplacian pyramids for each image
        lap_pyramids.append(build_pyramids(readImage, p_size))

jaccard_thresh = 0.3
for imidx, image in enumerate(lap_pyramids):
    image = cv.cvtColor(image[0], cv.COLOR_BGR2GRAY)
    for template in lap_templates:
        # Densely matches all template images with main image using the normalised correlation coefficient
        template = cv.cvtColor(template[0], cv.COLOR_BGR2GRAY)
        h, w = template.shape
        result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        cv.imshow("res", result)
        threshold = 0.9  # Threshold found by testing, 0.87 is optimal for pyramid size of 3 but 0.9 for size of 4
        loc = np.where(result >= threshold)
        # Gets topleft and botright points for each point above the threshold that has been found
        box_arrays = [[x_pt, y_pt, x_pt + w, y_pt + h, result[y_pt, x_pt]] for x_pt, y_pt in zip(*loc[::-1])]
        # casts it to an np array for parallel processing in our non maximum suppression function
        box_arrays = np.array(box_arrays)
        boxes = get_jaccard(box_arrays, jaccard_thresh)
        for box in boxes:
            if len(box) < 4:
                continue  # Handles unusual conditions in which a bounding box is incorrectly returned
            # Unpacks box points and casts each one to int
            x_pt, y_pt, x2_pt, y2_pt = map(int, box)
            # Adds a rectangle in the correct location for each found point
            cv.rectangle(colour_images[imidx], (x_pt * (2 ** p_size), y_pt * (2 ** p_size)),
                         (x2_pt * (2 ** p_size), y2_pt * (2 ** p_size)), (0, 0, 255), 1)


    cv.imshow('Display image', colour_images[imidx])
    cv.waitKey(0)

cv.destroyAllWindows()
