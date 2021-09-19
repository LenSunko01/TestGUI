from scipy.spatial import distance as dist
import numpy as np
import cv2
import sys
import argparse

fragment_size = 100


# accepts coordinates of lower left corner and returns picture's fragment with these coordinates
def get_image(coord_x, coord_y):
    fragment_mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.rectangle(fragment_mask, (coord_x, coord_y), (coord_x + fragment_size, coord_y + fragment_size), 255, -1)
    return cv2.bitwise_and(img, img, mask=fragment_mask)


def print_results(method_name, x1, y1, x2, y2):
    concat = np.concatenate((get_image(x1, y1), get_image(x2, y2)), axis=1)
    cv2.imshow(method_name, concat)


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
                    help="path to the image")
args = vars(parser.parse_args())

# histogram dictionary will contain histograms corresponding to each rectangular
histograms = {}

path = args["image"]
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if img is None:
    sys.exit("Could not read the image.")

cv2.imshow("Initial picture", img)
cv2.waitKey(0)

# get dimensions of image
dimensions = img.shape

# height and width of the image
height = img.shape[0]
width = img.shape[1]

for row in range(0, height - fragment_size, 10):
    for col in range(0, width - fragment_size, 10):
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (col, row), (col + fragment_size, row + fragment_size), 255, -1)
        masked = cv2.bitwise_and(img, img, mask=mask)
        histogram = cv2.calcHist([img], [0], mask, [256], [0, 256])
        histograms[(col, row)] = histogram

# initialize the scipy methods to compute distances
SCIPY_METHODS = (
    ("Euclidean", dist.euclidean),
    ("Manhattan", dist.cityblock))

# loop over the comparison methods
for (methodName, method) in SCIPY_METHODS:
    # initialize the results dictionary
    results = {}
    best = -1
    for coord1, hist1 in histograms.items():
        for coord2, hist2 in histograms.items():
            if coord1 == coord2:
                continue
            d = method(hist1, hist2)
            if d < best or best == -1:
                results[methodName] = (coord1, coord2)
                best = d

    (coord1, coord2) = results[methodName]
    print_results(methodName, coord1[0], coord1[1], coord2[0], coord2[1])
    cv2.waitKey(0)
