import cv2 as cv
import numpy as np
import camera

lower = 20
upper = 150


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
# cam = cv.VideoCapture(0)
cam, xi_img = camera.setup_camera()

cv.namedWindow('Camera')
cv.createTrackbar('Low', 'Camera', lower, 255, lower_clb)
cv.createTrackbar('Up', 'Camera', upper, 255, upper_clb)


def image_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    padding_height, padding_width = kernel_height // 2, kernel_width // 2
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')
    output_image = np.zeros_like(image)
    for y in range(image_height):
        for x in range(image_width):
            region = padded_image[y:y + kernel_height, x:x + kernel_width]
            output_image[y, x] = np.sum(region * kernel)
    return output_image


# Blur
blur = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
blur = blur * 1 / 32
# Edge detection
kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


def draw_contours(image, threshold_image):
    image[threshold_image != 0] = (0, 255, 0, 255)
    return image


while True:
    # _, img = cam.read()
    cam.get_image(xi_img)
    img = xi_img.get_image_data_numpy()
    img = cv.resize(img, (1280, 720))

    img_orig = img.copy()
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.filter2D(img, -1, blur)

    # Threshold image
    threshold_img = threshold(img, low_limit=lower, upper_limit=upper)
    cv.imshow("Threshold", cv.resize(threshold_img, (640, 480)))
    img = cv.filter2D(threshold_img, -1, kernel)

    # Mask
    # mask = np.zeros_like(img_orig[..., 2])
    #  B G R Alpha
    # mask[img_orig[..., 2] != 0] = (0, 255, 0, 255)
    # res = cv.addWeighted(img_orig, 1, mask, 1, 0)
    res = draw_contours(img_orig, img)
    cv.imshow('Camera', cv.resize(res, (640, 480)))
    pressed = cv.waitKey(1)
    if pressed == ord('q'):
        cv.destroyWindow('Camera')
        cv.destroyWindow('Threshold')
        break
    if pressed == ord('s'):
        cv.imwrite('contours-green.jpg', res)

img_custom = cv.cvtColor(img_orig, cv.COLOR_BGR2GRAY)
img_custom = image_convolution(img_custom, blur)
img_custom = threshold(img_custom, low_limit=lower, upper_limit=upper)
img_custom = image_convolution(img_custom, kernel)
img_custom = draw_contours(img_orig, img_custom)

horizontal_stack = np.hstack((cv.resize(img_custom, (320, 240)), cv.resize(res, (320, 240))))

cv.imshow("Results", horizontal_stack)

pressed = cv.waitKey(0)
cv.destroyAllWindows()
