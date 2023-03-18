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
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = threshold(img, low_limit=lower, upper_limit=upper)
    cv.imshow('Camera', img)
    pressed = cv.waitKey(1)
    if pressed == ord('q'):
        break

cv.destroyAllWindows()
