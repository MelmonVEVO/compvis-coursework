import os
import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
from timeit import default_timer
from csv import writer, QUOTE_MINIMAL

images_directory = "Task3AdditionalTestDataset/images"
templates_directory = "Task2Dataset/Training/png"
# change above as necessary

# read in all images and templates
template_images = []

for filename in os.listdir(templates_directory):
    template_filepath = os.path.join(templates_directory, filename)
    if os.path.isfile(template_filepath):
        im = cv2.imread(template_filepath)
        template_images.append(cv2.resize(im, (0, 0), fx=1 / 8, fy=1 / 8))

searchable_images = []

for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        readImage = cv2.imread(image_filepath)
        searchable_images.append(readImage)

# Compare each template against each test image
timings = []
min_matches = 9  # Minimum number of good feature matches to consider a true match
threshold = 250  # maximum distance threshold, change as appropriate

for image in searchable_images:
    for template in template_images:
        h, w = template.shape[:2]
        # Make brute force matcher for matching template features onto the main features
        bf = cv2.BFMatcher(
            cv2.NORM_L2, 
            crossCheck=True
        )
        # make a SIFT algorithm
        sift = cv2.SIFT_create(
            nOctaveLayers=6,
            edgeThreshold=15,
            contrastThreshold=0.08
        )  # contrastThreshold becomes less effective as nOctaveLayers increases
        # This means you must increase contrastThreshold proportionally to any increase in nOctaveLayers

        # create key points and descriptors for the main image and the template image
        # These are used to match the template on top of the image with scale invariant property
        key_points_template, descriptors_template = sift.detectAndCompute(template, None)
        key_points_main, descriptors_main = sift.detectAndCompute(image, None)

        # get matches sorted based on distance
        # (this is distance in brute force match space, so low distance is good match)
        matches = sorted(bf.match(descriptors_template, descriptors_main), key=lambda x: x.distance)

        good_matches = [found_match for found_match in matches if found_match.distance < threshold][:10]

        if len(good_matches) > min_matches:  # only run with enough good matches
            points = np.array(
                [[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]],  # Appropriately formatted like (-1, 1, 2)
                dtype=np.float32
            )  # Defines the key points of the image, top left, bot left, bot right, top right

            src_points = np.array(
                [key_points_template[match.queryIdx].pt for match in good_matches],
                dtype=np.float32
            ).reshape(-1, 1, 2)  # Finds the source points to draw from

            dest_points = np.array(
                [key_points_main[match.trainIdx].pt for match in good_matches],
                dtype=np.float32
            ).reshape(-1, 1, 2)  # Finds the destination points to draw to

            homography, mask = cv2.findHomography(
                src_points,
                dest_points,
                cv2.RANSAC,
                5.0
            )  # Uses RANSAC to map the source points to the destination points and find a homography
            matchesMask = list(mask.ravel())

            # Adds width offset to each point by adding coord vector in form (width, height)
            dest = cv2.perspectiveTransform(points, homography) + (w, 0)

            # Defines the way in which to draw the matches and starting points
            draw_params = {
                "matchColor": (0, 255, 0),  # Match colour
                "singlePointColor": None,
                "matchesMask": matchesMask,  # The matches that should be drawn
                "flags": 2
            }

            # Updates final image
            final_img = cv2.drawMatches(
                template,  # Template image drawn
                key_points_template,  # Key points on the template image
                image,  # Main image drawn
                key_points_main,  # Key points on the main image to be matched to
                good_matches,  # The subset of point pairs that are good matches
                None,
                **draw_params  # The unpacked dictionary values of drawing parameters to use for viewing
            )  # Draws found matches to final rendered image

            # Updates final image
            final_img = cv2.polylines(
                final_img,
                [np.array(dest, dtype=np.int32)],
                True,
                (0, 0, 255),  # Defines the line colours for the matches
                3,  # Line thickness
                cv2.LINE_AA  # Line draw type
            )  # Draws calculated bounding box to final image in red

            cv2.imshow("result", final_img)
            cv2.waitKey()

with open('Task3_times.csv', 'w') as csvfile:
    wt = writer(csvfile, quotechar='|', quoting=QUOTE_MINIMAL)
    wt.writerow(timings)
