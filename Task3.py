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
        template_images.append(cv2.resize(im, (0, 0), fx=1/8, fy=1/8))

searchable_images = []

for im_filename in os.listdir(images_directory):
    image_filepath = os.path.join(images_directory, im_filename)
    if os.path.isfile(image_filepath):
        readImage = cv2.imread(image_filepath)
        searchable_images.append(readImage)

# Compare each template against each test image
timings = []
for image in searchable_images:
    st = default_timer()
    for template in template_images:

        img1 = template
        img2 = image

        sift = cv2.SIFT_create()

        # create key points and descriptions
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # use brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(des1, des2) # get matches
        matches = sorted(matches, key = lambda x:x.distance) # sort matches based on distance
        threshold = 250 # maximum distance threshold, change as appropriate
        good = []
        for i in range(len(matches)):
            if matches[i].distance < threshold: # filter out all bad matches
                good.append(matches[i])

        if len(good) > 9: # only run with enough good matches
            good = good[:10] # take 10 best matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # builder matches mask
            matchesMask = mask.ravel().tolist()
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, M)
            dst += (w, 0)  # adding offset

            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

            # Draw bounding box in Red
            img3 = cv2.polylines(img3, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

            # cv2.imshow("result", img3)
            # cv2.waitKey()

    et = default_timer()
    timings.append(et - st)

with open('Task3_times.csv', 'w') as csvfile:
    wt = writer(csvfile, quotechar='|', quoting=QUOTE_MINIMAL)
    wt.writerow(timings)
