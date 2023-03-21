import cv2 as cv
import numpy as np

lower = 0
upper = 0


# Callbacks
def lower_clb(val):
    global lower
    lower = val


def upper_clb(val):
    global upper
    upper = val


# Algorithms
def threshold(image, low_limit, upper_limit):
    mask = np.logical_and(image > low_limit, image < upper_limit)
    image[mask] = 255
    image[~mask] = 0
    return image


# Main code
cam = cv.VideoCapture(0)
cv.namedWindow('Camera')
cv.createTrackbar('Low', 'Camera', 0, 255, lower_clb)
cv.createTrackbar('Up', 'Camera', 0, 255, upper_clb)

while True:
    _, img = cam.read()
    img_orig = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Blur
    blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    blur = blur * 1/32
    img = cv.filter2D(img, -1, blur)

    # Threshold image
    img = threshold(img, low_limit=lower, upper_limit=upper)

    # Edge detection
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    img = cv.filter2D(img, -1, kernel)

    # Mask
    mask = np.zeros_like(img_orig)
    mask[img != 0] = (0, 255, 0)
    res = cv.addWeighted(img_orig, 1, mask, 1, 0)

    cv.imshow('Camera', res)

    pressed = cv.waitKey(1)
    if pressed == ord('q'):
        break
    if pressed == ord('s'):
        cv.imwrite('contours.jpg', img)

cv.destroyAllWindows()
