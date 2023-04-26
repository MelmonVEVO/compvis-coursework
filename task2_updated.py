import os
import cv2 as cv
import numpy as np

# Edit these as needed such that they point to the correct directories
templates_directory = "Task2Dataset/Training/png"
images_directory = "Task2Dataset/TestWithoutRotations/images"
image_label_directory = "Task2Dataset/TestWithoutRotations/annotations"

p_size = 3  # Laplacian pyramid size
threshold = 0.825  # Threshold found by testing, 0.825 is optimal for pyramid size of 3 but 0.898 for others
jaccard_thresh = 0.1  # Threshold for jaccard index, should be low as we want minimal overlap


# Translated into NumPy from a similar function implemented with PyTorch at:
# https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/
def get_jaccard(coords: np.ndarray, thresh: float) -> list:
    """ Non-maximum suppression algorithm using the jaccard overlap index to eliminate multiple detections
    :param coords: list of all points and their confidence scores
    :param thresh: jaccard threshold
    :return: list of all scores above threshold
    """
    if coords.shape == (0,):  # No coordinates
        return []

    # Gets numpy arrays of each individual coordinate. x1y1 are the top left corner and x2y2 are bot right
    x1 = coords[:, 0].astype(float)
    y1 = coords[:, 1].astype(float)
    x2 = coords[:, 2].astype(float)
    y2 = coords[:, 3].astype(float)
    # Gets correlation coefficient of each box to ensure only the highest are kept
    # names = coords[:, 4]
    corr_coeffs = coords[:, 5].astype(float)


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
        x1_ord, y1_ord = x1[ordered_coeffs], y1[ordered_coeffs]
        x2_ord, y2_ord = x2[ordered_coeffs], y2[ordered_coeffs]
        # These are the union points for each main corner

        # Finds the intersection points
        x1_high, y1_high = np.maximum(x1_ord, x1[highest_ind]), np.maximum(y1_ord, y1[highest_ind])
        x2_high, y2_high = np.minimum(x2_ord, x2[highest_ind]), np.minimum(y2_ord, y2[highest_ind])

        # Parallel calculates the widths and heights of each intersection, with non intersecting set to 0
        widths = np.maximum(x2_high - x1_high, 0)
        heights = np.maximum(y2_high - y1_high, 0)

        # Finds intersection areas for each box
        inter_area = widths * heights

        # Finds union of areas of boxes. Will be lower for objects with an intersection.
        union_area = (box_orders - inter_area) + area_arr[highest_ind]

        # Calculates final jaccard score
        jaccard_scores = inter_area / union_area

        # Any records with scores too high (i.e. too much overlap) are discarded
        ordered_coeffs = ordered_coeffs[jaccard_scores < thresh]

    return final


def build_pyramids(img: np.ndarray, pyr_height: int = 3) -> list:
    temp_image = img.copy()
    temp_gauss = [temp_image]
    # Forms the gaussian pyramid, starting with the original image in the pyramid
    for i in range(pyr_height):
        # The pyrDown function uses a gaussian kernel on the image then downsamples to half its size by default
        temp_image = cv.pyrDown(temp_image)
        # Adds downsampled image to gauss pyramid
        temp_gauss.append(temp_image)

    return temp_gauss[::-1]


gauss_templates = []
template_names = []
for filename in os.listdir(templates_directory):  # scale by 1/8
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        im = cv.imread(template_filepath)
        gauss_templates.append(build_pyramids(cv.resize(im, (0, 0), fx=1 / 8, fy=1 / 8), p_size))
        template_names.append(filename[4:].split(".")[0])

colour_images = []
gauss_pyramids = []  # List of laplacian pyramids

for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        readImage = cv.imread(image_filepath)
        colour_images.append(readImage)
        # Adds the laplacian pyramids for each image
        gauss_pyramids.append(build_pyramids(readImage, p_size))

image_contains = []
for txt_filename in os.listdir(image_label_directory):
    text_filepath = os.path.join(image_label_directory, txt_filename)
    if os.path.isfile(text_filepath):
        holder = set()
        with open(text_filepath, "r") as t:
            for line in t:
                temp_text = line.split(", ")[0]
                holder.add(temp_text)
        image_contains.append(holder)

found_in_image = []
for imidx, image in enumerate(gauss_pyramids):
    # image = cv.cvtColor(image[0], cv.COLOR_BGR2GRAY)
    image = image[0]
    holder = set()
    box_arrays = []
    for template_name, template in zip(template_names, gauss_templates):
        # Densely matches all template images with main image using the normalised correlation coefficient
        # template = cv.cvtColor(template[0], cv.COLOR_BGR2GRAY)
        template = template[0]
        h, w = template.shape[:2]
        result = cv.matchTemplate(image, template, cv.TM_CCOEFF_NORMED)
        cv.imshow("res", result)
        loc = np.where(result >= threshold)
        # Gets topleft and botright points for each point above the threshold that has been found
        box_arrays.extend(  # Adds these points to an array which contains all possible matches for any template
            [[x_pt, y_pt, x_pt + w, y_pt + h, template_name, result[y_pt, x_pt]] for x_pt, y_pt in zip(*loc[::-1])]
        )

    box_arrays = np.array(box_arrays)  # Casts to a NP array for efficient operations
    boxes = get_jaccard(box_arrays, jaccard_thresh)  # Returns only correct bounding boxes after NMS
    for box in boxes:
        if len(box) < 5:
            continue  # For incorrectly formatted bounding boxes will just skip
        x_pt, y_pt, x2_pt, y2_pt = map(int, box[:-1])  # Find box pixel coords
        box_name = box[-1]
        holder.add(box_name)
        cv.rectangle(colour_images[imidx], (x_pt * (2 ** p_size), y_pt * (2 ** p_size)),
                     (x2_pt * (2 ** p_size), y2_pt * (2 ** p_size)), (0, 0, 255), 1)
        cv.putText(colour_images[imidx], box_name, (x_pt * (2 ** p_size), y_pt * (2 ** p_size) - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    found_in_image.append(holder)

    cv.imshow(f"Pyramid: {p_size}, Threshold: {threshold}", colour_images[imidx])
    cv.waitKey(0)

cv.destroyAllWindows()

true_positives = 0
false_positives = 0
false_negatives = 0
for found_template, actual_template in zip(found_in_image, image_contains):
    t_pos = found_template & actual_template
    true_positives += len(t_pos)
    f_pos = found_template - actual_template
    false_positives += len(f_pos)
    f_neg = actual_template - found_template
    false_negatives += len(f_neg)

print("Precision", true_positives / (true_positives + false_positives))
print("Recall", true_positives / (true_positives + false_negatives))
