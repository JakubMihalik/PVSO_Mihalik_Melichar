import cv2
import numpy as np
import cv2 as cv

global threshold1
global threshold2


def threshold_1_callback(val):
    global threshold1
    threshold1 = val


def threshold_2_callback(val):
    global threshold2
    threshold2 = val


cam = cv.VideoCapture(0)
window = cv.namedWindow('Camera')
cv.createTrackbar('Lower limit', 'Camera', 1, 255, threshold_1_callback)
cv.createTrackbar('Upper limit', 'Camera', 1, 255, threshold_2_callback)

while True:
    # Read image from camera
    _, img = cam.read()
    img_orig = img

    # Image processing
    # img = cv.Canny(img, threshold1, threshold2)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, threshold1, threshold2, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(image=img, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    img_copy = img_orig.copy()
    img_cont = cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

    # Show image
    cv.imshow('Camera', img)

    # Control logic
    pressed_key = cv.waitKey(1)
    if pressed_key == ord('q'):
        break
    if pressed_key == ord('s'):
        cv.imwrite('img_contours.jpg', img)

cv.destroyAllWindows()
