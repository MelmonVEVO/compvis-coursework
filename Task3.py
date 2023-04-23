import os
import numpy as np
import matplotlib, cv2
import matplotlib.pyplot as plt
from timeit import default_timer
from csv import reader, writer, QUOTE_MINIMAL


def area_intersect(rect1, rect2):
    x1 = max(rect1[0][0], rect2[0][0])
    y1 = max(rect1[0][1], rect2[0][1])
    x2 = min(rect1[1][0], rect2[1][0])
    y2 = min(rect1[1][1], rect2[1][1])

    # Calculate the intersection area
    if x1 < x2 and y1 < y2:
        return (x2 - x1) * (y2 - y1)
    else:
        return 0

def area_union(rect1, rect2):
    x1 = min(rect1[0][0], rect2[0][0])
    y1 = min(rect1[0][1], rect2[0][1])
    x2 = max(rect1[1][0], rect2[1][0])
    y2 = max(rect1[1][1], rect2[1][1])

    # Calculate the union area
    return (x2 - x1) * (y2 - y1)

def read_annotaiton(annot):
    with open(annot) as f:
        rd = reader(f)
        w=[]
        for row in rd:
            w.extend(row)

    box1 = ((int(w[1][2:]), int(w[2][1:-1])), (int(w[3][2:]), int(w[4][1:-1])))
    box2 = ((int(w[6][2:]), int(w[7][1:-1])), (int(w[8][2:]), int(w[9][1:-1])))
    box3 = ((int(w[11][2:]), int(w[12][1:-1])), (int(w[13][2:]), int(w[14][1:-1])))
    box4 = ((int(w[16][2:]), int(w[17][1:-1])), (int(w[18][2:]), int(w[19][1:-1])))
    if len(w) > 20:
        box5 = ((int(w[21][2:]), int(w[22][1:-1])), (int(w[23][2:]), int(w[24][1:-1])))
    else:
        box5 = ()

    return box1, box2, box3, box4, box5

images_directory = "Task3AdditionalTestDataset/images"
templates_directory = "Task2Dataset/Training/png"
annotations_directory = "Task3AdditionalTestDataset/annotations"
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

annotations = []
for an_filename in os.listdir(annotations_directory):
    annotation_filepath = os.path.join(annotations_directory, an_filename)
    if os.path.isfile(annotation_filepath):
        annotations.append(annotation_filepath)

# Compare each template against each test image
timings = []
min_matches = 9  # Minimum number of good feature matches to consider a true match
threshold = 250  # maximum distance threshold, change as appropriate
tp = 0
fp = 0
neg = 0
exp_pos = 0

sift = cv2.SIFT_create(
    nOctaveLayers=5,
    edgeThreshold=15,
    contrastThreshold=0.05
)  # contrastThreshold becomes less effective as nOctaveLayers increases

for image, annotation in zip(searchable_images, annotations):
    # Create key points and descriptors for main image
    rect1, rect2, rect3, rect4, rect5 = read_annotaiton(annotation)
    if len(rect5) > 1:
        exp_pos += 5
    else:
        exp_pos += 4
    key_points_main, descriptors_main = sift.detectAndCompute(image, None)
    for template in template_images:
        h, w = template.shape[:2]
        # use brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        # make a SIFT algorithm
        # This means you must increase contrastThreshold proportionally to any increase in nOctaveLayers

        # create key points and descriptors for the template image
        # These are used to match the template on top of the image with scale invariant property
        key_points_template, descriptors_template = sift.detectAndCompute(template, None)

        # get matches sorted based on distance
        # (this is distance in brute force match space, so low distance is good match)
        matches = sorted(bf.match(descriptors_template, descriptors_main), key=lambda x: x.distance)

        good_matches = [found_match for found_match in matches if found_match.distance < threshold][:10]

        if len(good_matches) > min_matches:  # only run with enough good matches
            points = np.array(
                [[[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]],  # Appropriately formatted as points
                dtype=np.float32  # This formatting is done so minimal indexes need to be used for arithmetic operations
            )  # Defines the key points of the image, top left, bot left, bot right, top right

            src_points = np.array(
                [key_points_template[match.queryIdx].pt for match in good_matches],  # Finds the loc of each source
                dtype=np.float32  # For each good match (this is in the template image)
            ).reshape(-1, 1, 2)  # Finds the source points to draw from, apropriately reformatted for minimal indexing

            dest_points = np.array(
                [key_points_main[match.trainIdx].pt for match in good_matches],  # Finds the loc of each good match
                dtype=np.float32  # In the destination image (i.e. main image)
            ).reshape(-1, 1, 2)  # Finds the destination points to draw to, appropriately reformatted for min indexing

            homography, mask = cv2.findHomography(
                src_points,  # Finds the homography between the template image points and the main image points
                dest_points,
                cv2.RANSAC,  # Uses random sample consensus to eliminate any noise in the scale
                5.0  # Ransac size
            )  # Uses RANSAC to map the source points to the destination points and find a homography
            matchesMask = list(mask.ravel())  # This essentially converts the np array from n dimensions into 1D list

            # Adds width offset to each point by adding coord vector in form (width, height)
            # The width offset is needed as we display the template image next to the main image
            # The perspective transform maps the homography points found to the actual points in the image
            dest = cv2.perspectiveTransform(points, homography)

            x, y, w, h = cv2.boundingRect(dest)
            rect = (x, y), (x+w, y+h)
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 0), 2)

            # Defines the way in which to draw the matches and starting points
            draw_params = {
                "matchColor": (0, 255, 0),  # Match colour
                "singlePointColor": None,  # No need to colour points individually without a match
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

            #calculate intersection of unions for each ground-truth box
            IoU1 = area_intersect(rect1, rect) / area_union(rect1, rect)
            IoU2 = area_intersect(rect2, rect) / area_union(rect2, rect)
            IoU3 = area_intersect(rect3, rect) / area_union(rect3, rect)
            IoU4 = area_intersect(rect4, rect) / area_union(rect4, rect)
            if len(rect5) > 1:
                IoU5 = area_intersect(rect5, rect) / area_union(rect5, rect)
            else:
                IoU5 = 0
            if IoU1 > 0.5:
                tp+=1
            elif IoU2 > 0.5:
                tp+=1
            elif IoU3 > 0.5:
                tp+=1
            elif IoU4 > 0.5:
                tp+=1
            elif IoU5 > 0.5:
                tp+=1
            else:
                fp+=1

            cv2.imshow("result", final_img)
            cv2.waitKey()

        else:
            neg += 1


fneg = exp_pos - (tp + fp)

with open('Task3_stats.csv', 'w') as csvfile:
    wt = writer(csvfile, quotechar='|', quoting=QUOTE_MINIMAL)
    wt.writerow(["Threshold:" + str(threshold)])
    wt.writerow(["nOctaveLayers:" + str(5)])
    wt.writerow(["edgeThreshold:" + str(15)])
    wt.writerow(["contrastThreshold:" + str(0.05)])
    wt.writerow(["True Positives:" + str(tp)])
    wt.writerow(["False Positives:" + str(fp)])
    wt.writerow(["Precision:" + str(tp/(tp+fp))])
    wt.writerow(["Recall:" + str(tp/(tp+fneg))])
